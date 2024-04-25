#Deep deterministic policy gradient 
#Learn both the policy and Q-learning; used for continuous action space
#docs: https://spinningup.openai.com/en/latest/algorithms/ddpg.html
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

@dataclass 
class Args: 
    seed: int = 1 
    env_id: str = "Hopper-v4"
    total_timesteps: int = 500000 
    learning_rate : float = 3e-4 
    num_envs: int = 1 #number of parallel game envs 
    buffer_size: int = int(1e-6)
    gamma: float = 0.99 #Discount factor 
    tau: float = 0.05  #Target network update rate 
    target_network_frequency: int = 500 #Timesteps it takes to update the target network 
    batch_size: int = 256 
    exploration_noise : float = 0.1 
    #start_e: float= 1 #Epsilon: Exploration should be large in the beginning 
    #end_e: float = 0.05 #Ending epsilon: Exploration rate 
    #exploration_fraction:  float = 0.5
    learning_starts: int = 25e3 #Learning starts each 10k steps 
    policy_frequency: int = 2 
    noise_clip: float = 0.5 #Noise clip param of the target policy smoothing regularization 

def make_env(env_id, seed,idx): 
    def thunk(): 
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env 
    return thunk 

class QNetwork(nn.Module): 
    @nn.compact 
    def __call__(self, x: jnp.ndarray, a: jnp.array): 
        x = jnp.concatenate([x,a], -1) #In the Q network, itput arrays of state and actions 
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x) #return final action 
        return x 

class Actor(nn.Module): #Policy function, map states to actions 
    action_dim: int  
    action_scale: jnp.array 
    action_bias: jnp.array 

    @nn.compact 
    def __call__(self, x): 
        x = nn.Dense(256)(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x) #Probs over possible next actions 
        x = nn.tanh(x)
        x = x * self.action_scale + self.action_bias #Scale to desired action range 
        return x 
    
class TrainState(TrainState): 
    target_params: flax.core.FrozenDict 

if __name__ == "__main__": 
    args = tyro.cli(Args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, actor_key, qf1_key = jax.random.split(key, 3)

    #Setting up envs 
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "Only continuous action space is allowed"
    max_action = float(envs.single_action_space.high[0])
    envs.single_observation_space.dtype = np.float32 

    rb = ReplayBuffer(
        args.buffer_size, envs.single_observation_space, envs.single_action_space, 
        device = "cpu", handle_timeout_termination= False
    )
    #Start the game 
    obs, _ = envs.reset(seed = args.seed)
    actor = Actor(
        action_dim = np.prod(envs.single_action_space.shape), 
        action_scale = jnp.array((envs.action_space.high - envs.action_space.low) / 2.0), 
        action_bias = jnp.array((envs.action_space.high + envs.action_space.low) / 2.0)

    )
    actor_state = TrainState.create(
        apply_fn = actor.apply, params = actor.init(actor_key,obs), 
        target_params = actor.init(actor_key,obs), 
        tx = optax.adam(learning_rate = args.learning_rate)
    )
    qf = QNetwork()
    qf1_state = TrainState.create(
        apply_fn = qf.apply, 
        params = qf.init(qf1_key, obs, envs.action_space.sample()), 
        target_params = qf.init(qf1_key, obs, envs.action_space.sample()), 
        tx = optax.adam(learning_rate = args.learning_rate)
    )
    actor.apply = jax.jit(actor.apply)
    qf.apply = jax.jit(qf.apply)

    @jax.jit 
    #minimize q value loss 
    def update_critic(
        actor_state: TrainState, 
        qf1_state: TrainState, 
        observations: np.ndarray,
        actions: np.ndarray, 
        next_observations: np.ndarray, 
        rewards: np.ndarray, terminations: np.ndarray,  
    ): #Updating value function network 
        next_state_actions = (actor.apply(actor_state.target_params, next_observations)).clip(-1,1)
        qf1_next_target = qf.apply(qf1_state.target_params, next_observations, next_state_actions).reshape(-1) #Target q values for next state actions 
        next_q_value = (rewards + (1-terminations) * args.gamma * (qf1_next_target)).reshape(-1) #Find q value using Bellman equation 
        def mse_loss(params): 
            qf_a_values = qf.apply(params, observations, actions).squeeze() #current q value
            return ((qf_a_values - next_q_value) ** 2).mean(), qf_a_values.mean()
        (qf1_loss_value, qf1_a_values), grads1 = jax.value_and_grad(mse_loss, has_aux=True)(qf1_state.params)
        qf1_state = qf1_state.apply_gradients(grads = grads1) #update 
        return qf1_state, qf1_loss_value, qf1_a_values 
    
    @jax.jit 
    #maximize returns 
    def update_actor(
        actor_state: TrainState, qf1_state: TrainState, 
        observations: np.ndarray 
    ): 
        def actor_loss(params): 
            return -qf.apply(qf1_state.params, observations, actor.apply(params, observations)).mean()
        actor_loss_value, grads = jax.value_and_grad(actor_loss)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)
        actor_state = actor_state.replace(
            target_params = optax.incremental_update(actor_state.params, actor_state.target_params, args.tau)
        )
        #update critic network 
        qf1_state = qf1_state.replace(
            target_params = optax.incremental_update(qf1_state.params, qf1_state.target_params, args.tau)

        )
        return actor_state, qf1_state, actor_loss_value 
    
    start_time = time.time()
    for step in range(args.total_timesteps): 
        if step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions = actor.apply(actor_state.params, obs)
            actions = np.array(
                [
                    (jax.device_get(actions)[0] + np.random.normal(0, actor.action_scale * args.exploration_noise)[0]).clip(
                        envs.single_action_space.low, envs.single_action_space.high
                    )
                ]
            )
        next_obs, rewards,terminations, truncations, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={step}, episodic_return={info['episode']['r']}")

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        #Add data to replay buffer 
        rb.add(obs, real_next_obs, actions, rewards, terminations,infos)
        obs = next_obs 

        #Training 
        if step > args.learning_starts: 
            data = rb.sample(args.batch_size)
            qf1_state, qf1_loss_value, qf1_a_values = update_critic(
                actor_state, qf1_state, data.observations.numpy(), 
                data.actions.numpy(),
                data.next_observations.numpy(),
                data.rewards.flatten().numpy(),
                data.dones.flatten().numpy(),
            )
            if step % args.policy_frequency == 0: 
                actor_state, qf1_state, actor_loss_value = update_actor(
                    actor_state, qf1_state, data.observations.numpy()
                )
            if step % 100 == 0: 
                print("SPS:", int(step / (time.time() - start_time)))
    envs.close()
