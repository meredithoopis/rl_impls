import os 
import random 
import time 
from dataclasses import dataclass 
import flax 
import flax.linen as nn 
import gymnasium as gym 
import jax 
import jax.numpy as jnp 
import numpy as np 
import optax 
import tyro 
from flax.training.train_state import TrainState 
from stable_baselines3.common.buffers import ReplayBuffer 
import stable_baselines3 as sb3 

@dataclass 
class Args: 
    seed: int = 1 
    env_id: str = "CartPole-v1"
    total_timesteps: int = 500000 
    learning_rate : float = 2.5e-4 
    num_envs: int = 1 #number of parallel game envs 
    buffer_size: int = 10000
    gamma: float = 0.99 #Discount factor 
    tau: float = 1.0 #Target network update rate 
    target_network_frequency: int = 500 #Timesteps it takes to update the target network 
    batch_size: int = 128 
    start_e: float= 1 #Epsilon: Exploration should be large in the beginning 
    end_e: float = 0.05 #Ending epsilon: Exploration rate 
    exploration_fraction:  float = 0.5
    learning_starts: int = 10000 #Learning starts each 10k steps 
    train_frequency: int = 10 


def make_env(env_id, seed,idx): 
    def thunk(): 
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env 
    return thunk 

class QNetwork(nn.Module): 
    action_dim: int   
    @nn.compact 
    def __call__(self,x: jnp.ndarray): 
        x = nn.Dense(120)(x)
        x = nn.relu(x)
        x = nn.Dense(84)(x)
        x = nn.relu(x)
        n = nn.Dense(self.action_dim)(x) #For DQN, final layer return probs over actions:Which action to take 
        return x 

class TrainState(TrainState): 
    target_params: flax.core.FrozenDict #Storing model params in an immutable way 

def linear_schedule(start_e: float, end_e: float, duration:int, t:int): 
    #compute the amount of exploration for a certain step using a linear function 
    slope = (end_e - start_e) / duration 
    return max(slope * t + start_e, end_e)


if __name__ == "__main__": 
    args = tyro.cli(Args)
    #assert args.num_envs == 1, "Not possible to have vectorized envs"
    #Seeding 
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(0)
    key, q_key = jax.random.split(key, 2)
    #Creating envs 
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i) for i in range(args.num_envs)]
    )
    #assert isinstance(envs.single_action_space, gym.spaces.Discrete), "Only discrete action space is possible"
    
    obs, _ = envs.reset(seed = args.seed)
    q_network = QNetwork(action_dim = envs.single_action_space.n)
    q_state = TrainState.create(
        apply_fn = q_network.apply, 
        params = q_network.init(q_key, obs), 
        target_params = q_network.init(q_key, obs), 
        tx = optax.adam(learning_rate = args.learning_rate)
    )
    q_network.apply = jax.jit(q_network.apply)
    q_state = q_state.replace(target_params = optax.incremental_update(q_state.params, q_state.target_params,1))
    
    rb = ReplayBuffer(
        args.buffer_size, envs.single_observation_space, envs.single_action_space, "cpu", handle_timeout_termination=False
    )

    @jax.jit
    def update(q_state, observations, actions, next_observations, rewards, dones):
        q_next_target = q_network.apply(q_state.target_params, next_observations)  # (batch_size, num_actions)
        q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
        next_q_value = rewards + (1 - dones) * args.gamma * q_next_target

        def mse_loss(params):
            q_pred = q_network.apply(params, observations)  # (batch_size, num_actions)
            q_pred = q_pred[jnp.arange(q_pred.shape[0]), actions.squeeze()]  # (batch_size,)
            return ((q_pred - next_q_value) ** 2).mean(), q_pred

        (loss_value, q_pred), grads = jax.value_and_grad(mse_loss, has_aux=True)(q_state.params)
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, q_pred, q_state
    start_time = time.time()

    #Start the game 
    obs, _ = envs.reset(seed = args.seed)
    for step in range(args.total_timesteps): 
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network.apply(q_state.params, obs)
            #actions = q_values.argmax(axis=-1)
            actions = q_values.argmax(axis=-1) % envs.single_action_space.n
            actions = jax.device_get(actions)
        #next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        if "final_info" in infos: 
            for info in infos['final_info']: 
                if info and "episode" in info: 
                    print(f"Global step={step}, episodic_return={info['episode']['r']}")

        #Save data to replay buffer 
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations): 
            if trunc: 
                real_next_obs[idx] = infos["final_observation"][idx]

            rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
            obs = next_obs 
            
            if step > args.learning_starts: 
                if step % args.train_frequency == 0: 
                    data = rb.sample(args.batch_size)
                    #A gradient descent step 
                    loss, old_val, q_state = update(
                        q_state, data.observations.numpy(), 
                        data.actions.numpy(), data.next_observations.numpy(), 
                        data.rewards.flatten().numpy(), data.dones.flatten().numpy()
                    )
                    if step % 100 == 0: 
                        print("SPS", int(step / (time.time() - start_time)))
                if step % args.target_network_frequency == 0: 
                    q_state = q_state.replace(
                        target_params = optax.incremental_update(q_state.params,q_state.target_params, args.tau)

                    )

    envs.close()

