# docs and experiment results can be found at   
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Sequence, Callable

os.environ[
    "XLA_PYTHON_CLIENT_MEM_FRACTION"
] = "0.7" 

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
from huggingface_hub import hf_hub_download
from rich.progress import track
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: nn.Module,
    epsilon: float = 0.05,
    capture_video: bool = False,
    seed=1,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0,capture_video,run_name)])
    obs, _ = envs.reset()
    model = Model(action_dim=envs.single_action_space.n)
    q_key = jax.random.PRNGKey(seed)
    params = model.init(q_key, obs)
    with open(model_path, "rb") as f:
        params = flax.serialization.from_bytes(params, f.read())
    model.apply = jax.jit(model.apply)

    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = model.apply(params, obs)
            actions = q_values.argmax(axis=-1)
            actions = jax.device_get(actions)
        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns

class TeacherModel(nn.Module):  #Teacher model to transfer knowledge to 
    action_dim: int  
    @nn.compact 
    def __call__(self, x): 
        x = jnp.transpose(x, (0,2,3,1))
        x = x / (255.0)
        x = nn.Conv(32, kernel_size=(8,8), strides=(4,4), padding="VALID")(x) #No padding needed
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4,4), strides=(2,2), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3,3), strides=(1,1), padding="VALID")(x) 
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x  #Return probs over possible actions 


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""

    env_id: str = "BreakoutNoFrameskip-v4"
    total_timesteps: int = 500000
    learning_rate: float = 1e-4
    num_envs: int = 1
    buffer_size: int = 10000
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 1000
    batch_size: int = 32
    start_e: float = 1.0
    end_e: float = 0.01
    exploration_fraction: float = 0.10

    learning_starts: int = 8000
    train_frequency: int = 4
    teacher_policy_hf_repo: str = None
    teacher_model_exp_name: str = "dqn_atari_jax"
    teacher_eval_episodes: int = 10
    teacher_steps: int = 50000
    offline_steps: int = 50000
    temperature: float = 1.0



def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(seed)

        return env

    return thunk

class ResidualBlock(nn.Module): 
    channels: int   
    @nn.compact 
    def __call__(self,x): 
        inputs = x 
        x = nn.relu(x)
        x = nn.Conv(
            self.channels, kernel_size= (3,3)
        )(x)
        x = nn.relu(x)
        x = nn.Conv(self.channels, kernel_size=(3,3))(x)
        return x + inputs 

class ConvSequence(nn.Module): 
    channels: int   
    @nn.compact 
    def __call__(self,x): 
        x = nn.Conv(
            self.channels, kernel_size = (3,3)
        )(x)
        x = nn.max_pool(x, window_shape = (3,3), strides = (2,2), padding='SAME')
        x = ResidualBlock(self.channels)(x)
        x = ResidualBlock(self.channels)(x)
        return x 
    
class QNetwork(nn.Module): 
    action_dim: int  
    channelss: Sequence[int] = (16,32,32)
    @nn.compact 
    def __call__(self,x): 
        x = jnp.transpose(x, (0,2,3,1))
        x = x / (255.0)
        for channels in self.channelss: 
            x = ConvSequence(channels)(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x #possible next actions 


class TrainState(TrainState): 
    target_params: flax.core.FrozenDict 

def linear_schedule(start_e: float, end_e: float, duration:int, t:int): 
    #compute the amount of exploration for a certain step using a linear function 
    slope = (end_e - start_e) / duration 
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    if args.teacher_policy_hf_repo is None:
        args.teacher_policy_hf_repo = f"cleanrl/{args.env_id}-{args.teacher_model_exp_name}-seed1"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, q_key = jax.random.split(key, 2)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(channelss=(16, 32, 32), action_dim=envs.single_action_space.n)

    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_network.init(q_key, envs.observation_space.sample()),
        target_params=q_network.init(q_key, envs.observation_space.sample()),
        tx=optax.adam(learning_rate=args.learning_rate),
    )
    q_network.apply = jax.jit(q_network.apply)
    
    teacher_model_path = hf_hub_download(
        repo_id=args.teacher_policy_hf_repo, filename=f"{args.teacher_model_exp_name}.cleanrl_model"
    )
    teacher_model = TeacherModel(action_dim=envs.single_action_space.n)
    teacher_model_key = jax.random.PRNGKey(args.seed)
    teacher_params = teacher_model.init(teacher_model_key, envs.observation_space.sample())
    with open(teacher_model_path, "rb") as f:
        teacher_params = flax.serialization.from_bytes(teacher_params, f.read())
    teacher_model.apply = jax.jit(teacher_model.apply)

    # evaluate the teacher model
    teacher_episodic_returns = evaluate(
        teacher_model_path,
        make_env,
        args.env_id,
        eval_episodes=args.teacher_eval_episodes,
        run_name=f"{run_name}-teacher-eval",
        Model=TeacherModel,
        epsilon=0.05,
        capture_video=False,
    )
    teacher_rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        "cpu",
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )

    obs, _ = envs.reset(seed=args.seed)
    for global_step in track(range(args.teacher_steps), description="filling teacher's replay buffer"):
        epsilon = linear_schedule(args.start_e, args.end_e, args.teacher_steps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = teacher_model.apply(teacher_params, obs)
            actions = q_values.argmax(axis=-1)
            actions = jax.device_get(actions)
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncated):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        teacher_rb.add(obs, real_next_obs, actions, rewards, terminated, infos)
        obs = next_obs

    def kl_divergence_with_logits(target_logits, prediction_logits):
        #on-policy distillation loss
        out = -nn.softmax(target_logits) * (nn.log_softmax(prediction_logits) - nn.log_softmax(target_logits))
        return jnp.sum(out)

    @jax.jit
    def update(q_state, observations, actions, next_observations, rewards, dones, distill_coeff):
        q_next_target = q_network.apply(q_state.target_params, next_observations)  # (batch_size, num_actions)
        q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
        td_target = rewards + (1 - dones) * args.gamma * q_next_target
        teacher_q_values = teacher_model.apply(teacher_params, observations)

        def loss(params, td_target, teacher_q_values, distill_coeff):
            student_q_values = q_network.apply(params, observations)  # (batch_size, num_actions)
            q_pred = student_q_values[np.arange(student_q_values.shape[0]), actions.squeeze()]  # (batch_size,)
            q_loss = ((q_pred - td_target) ** 2).mean()
            teacher_q_values = teacher_q_values / args.temperature
            student_q_values = student_q_values / args.temperature
            distill_loss = jnp.mean(jax.vmap(kl_divergence_with_logits)(teacher_q_values, student_q_values))
            overall_loss = q_loss + distill_coeff * distill_loss
            return overall_loss, (q_loss, q_pred, distill_loss)

        (loss_value, (q_loss, q_pred, distill_loss)), grads = jax.value_and_grad(loss, has_aux=True)(
            q_state.params, td_target, teacher_q_values, distill_coeff
        )
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, q_loss, q_pred, distill_loss, q_state

    # offline training phase: train the student model using the qdagger loss
    for global_step in track(range(args.offline_steps), description="offline student training"):
        data = teacher_rb.sample(args.batch_size)
        loss, q_loss, old_val, distill_loss, q_state = update(
            q_state,
            data.observations.numpy(),
            data.actions.numpy(),
            data.next_observations.numpy(),
            data.rewards.flatten().numpy(),
            data.dones.flatten().numpy(),
            1.0,
        )

        # update the target network
        if global_step % args.target_network_frequency == 0:
            q_state = q_state.replace(target_params = optax.incremental_update(q_state.params, q_state.target_params, args.tau))

        if global_step % 100000 == 0:
            # evaluate the student model
            model_path = f"runs/{run_name}/{args.exp_name}-offline-{global_step}.cleanrl_model"
            with open(model_path, "wb") as f:
                f.write(flax.serialization.to_bytes(q_state.params))
            print(f"model saved to {model_path}")

            episodic_returns = evaluate(
                model_path,
                make_env,
                args.env_id,
                eval_episodes=10,
                run_name=f"{run_name}-eval",
                Model=QNetwork,
                epsilon=0.05,
            )
            print(episodic_returns)

    rb = ReplayBuffer(

        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        "cpu",
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    obs, _ = envs.reset(seed=args.seed)
    episodic_returns = deque(maxlen=10)
    # online training 
    for global_step in track(range(args.total_timesteps), description="online student training"):
        global_step += args.offline_steps

        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network.apply(q_state.params, obs)
            actions = q_values.argmax(axis=-1)
            actions = jax.device_get(actions)

        next_obs, rewards, terminated, truncated, infos = envs.step(actions)

 
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if "episode" not in info:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                episodic_returns.append(info["episode"]["r"])
                break
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncated):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminated, infos)

        obs = next_obs

        # training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)

                if len(episodic_returns) < 10:
                    distill_coeff = 1.0
                else:
                    distill_coeff = max(1 - np.mean(episodic_returns) / np.mean(teacher_episodic_returns), 0)
                loss, q_loss, old_val, distill_loss, q_state = update(
                    q_state,
                    data.observations.numpy(),
                    data.actions.numpy(),
                    data.next_observations.numpy(),
                    data.rewards.flatten().numpy(),
                    data.dones.flatten().numpy(),
                    distill_coeff,
                )

                if global_step % 100 == 0:
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    print(distill_coeff)

            # update the target network
            if global_step % args.target_network_frequency == 0:
                q_state = q_state.replace(
                    target_params=optax.incremental_update(q_state.params, q_state.target_params, args.tau)
                )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(flax.serialization.to_bytes(q_state.params))
        print(f"model saved to {model_path}")


        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            epsilon=0.05,
        )


        if args.upload_model:
            from utils import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "Qdagger", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    