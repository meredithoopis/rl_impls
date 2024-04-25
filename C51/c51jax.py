#Ref: https://adityauser.github.io/posts/2019/06/C51/
import os
import random
import time
from dataclasses import dataclass
import flax 
import flax.linen as nn 
import gymnasium as gym 
import jax 
import jax.numpy as jnp 
import optax 
import tyro 
import numpy as np 
from flax.training.train_state import TrainState 
from stable_baselines3.common.buffers import ReplayBuffer 

@dataclass 
class Args: 
    capture_video: bool = True
    seed: int = 1 
    env_id: str = "CartPole-v1"
    total_timesteps: int = 500000 
    learning_rate : float = 2.5e-4 
    num_envs: int = 1 #number of parallel game envs 
    buffer_size: int = 10000
    n_atoms: int = 101 #Number of atomsL Possible values that future discounted values can take 
 
    v_min: float = -100 #return lower bound  
    v_max: float = 100 
    gamma: float = 0.99 #Discount factor 
    #tau: float = 0.05  #Target network update rate
    target_network_frequency: int = 500 #Timesteps it takes to update the target network 
    batch_size: int = 128
    #policy_noise: float = 0.2 
    #exploration_noise : float = 0.1 
    start_e: float= 1 #Epsilon: Exploration should be large in the beginning 
    end_e: float = 0.05 #Ending epsilon: Exploration rate 
    exploration_fraction:  float = 0.5
    learning_starts: int = 10000 #Learning starts each 10k steps 
    train_frequency: int = 10 
    #policy_frequency: int = 2 
    #noise_clip: float = 0.5 #Noise clip param of the target policy smoothing regularization 

def make_env(env_id, seed,idx, capture_video, run_name): 
    def thunk(): 
        if capture_video and idx == 0: 
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else: 
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env 
    return thunk 

class QNetwork(nn.Module): 
    # returns (PMFs) of the return distributions for each action
    action_dim: int  
    n_atoms: int  
    @nn.compact 
    def __call__(self, x): 
        x = nn.Dense(120)(x)
        x = nn.relu(x)
        x = nn.Dense(84)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim * self.n_atoms)(x)
        x = x.reshape((x.shape[0], self.action_dim, self.n_atoms))
        x = nn.softmax(x, axis = -1) #return pmfs of return 
        return x 

class TrainState(TrainState): 
    target_params: flax.core.FrozenDict 
    atoms: jnp.ndarray 

def linear_schedule(start_e: float, end_e: float, duration:int, t:int): 
    #compute the amount of exploration for a certain step using a linear function 
    slope = (end_e - start_e) / duration 
    return max(slope * t + start_e, end_e)

if __name__ == "__main__": 
    
    args = tyro.cli(Args)
    assert args.num_envs == 1 , "Vectorized envs are not supported"
    run_name = f"{args.env_id}_{int(time.time())}"
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, q_key = jax.random.split(key, 2)

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed+i, i, args.capture_video, run_name) for i in range(args.num_envs)] 
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "Action space should be discrete"

    obs, _ = envs.reset(seed = args.seed)
    q_network = QNetwork(action_dim= envs.single_action_space.n, n_atoms=args.n_atoms)
    q_state = TrainState.create(
        apply_fn=q_network.apply, 
        params = q_network.init(q_key,obs), 
        target_params = q_network.init(q_key, obs), 
        atoms = jnp.asarray(np.linspace(args.v_min, args.v_max, num = args.n_atoms)),  #to have a distributions over returns 
        tx = optax.adam(learning_rate=args.learning_rate, eps=0.01 / args.batch_size)
    )
    q_network.apply = jax.jit(q_network.apply)
    #q_state = q_state.replace(target_params=optax.incremental_update(q_state.params, q_state.target_params, 1))
    rb = ReplayBuffer(
        args.buffer_size, envs.single_observation_space, 
        envs.single_action_space, "cpu", handle_timeout_termination=False 
    )

    @jax.jit 
    def update(q_state, observations, actions, next_observations, rewards,dones): 
        next_pmfs = q_network.apply(q_state.target_params, next_observations) #(bs, num_actions, num_atoms)
        next_vals = (next_pmfs * q_state.atoms).sum(axis = -1) #(bs,num_actions): cal the E of the return dist for each action in the next state 

        next_action = jnp.argmax(next_vals, axis = -1) #(bs): select the action with the highest E
        next_pmfs = next_pmfs[np.arange(next_pmfs.shape[0]), next_action] #Extract next pmfs
        next_atoms = rewards + args.gamma * q_state.atoms * (1-dones) #target atom values for current s-a pairs using Bellman: potential future discounted returns 


        #Projection 
        delta_z = q_state.atoms[1] - q_state.atoms[0] #Spacing between atoms 
        tz = jnp.clip(next_atoms, a_min = (args.v_min), a_max = (args.v_max)) #Clip atom values 
        b = (tz-args.v_min)/ delta_z #Fractional index of the target atom values within the current set of atoms 
        l = jnp.clip(jnp.floor(b), a_min=0, a_max = args.n_atoms-1) #lower index 
        u = jnp.clip(jnp.ceil(b), a_min=0, a_max = args.n_atoms-1) #upper index 
        d_m_l = (u + (l==u).astype(jnp.float32) -b ) * next_pmfs #how much of the next state pmfs should be distributed to lower and upper atom indices 
        d_m_u = (b-l) * next_pmfs  
        target_pmfs = jnp.zeros_like(next_pmfs)
        def project_to_bins(i, val):
            #Distrubute next state pmfs onto current set of atoms  
            val = val.at[i, l[i].astype(jnp.int32)].add(d_m_l[i])
            val = val.at[i, u[i].astype(jnp.int32)].add(d_m_u[i])
            return val 
        target_pmfs = jax.lax.fori_loop(0, target_pmfs.shape[0], project_to_bins, target_pmfs) 

        def loss(q_params, observations, actions, target_pmfs): #Cross-entropy 
            pmfs = q_network.apply(q_params, observations)
            old_pmfs = pmfs[np.arange(pmfs.shape[0]), actions.squeeze()]
            old_pmfs_l = jnp.clip(old_pmfs, a_min = 1e-5, a_max = 1 - 1e-5)
            loss = (-(target_pmfs * jnp.log(old_pmfs_l)).sum(-1)).mean()
            return loss, (old_pmfs * q_state.atoms).sum(-1)
         
        (loss_value, old_values), grads = jax.value_and_grad(loss, has_aux=True)(
            q_state.params, observations, actions, target_pmfs
        )
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, old_values, q_state 
    
    start_time = time.time()
    obs,_ = envs.reset(seed = args.seed)
    for step in range(args.total_timesteps): 
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, step)
        if random.random() < epsilon: 
            actions = np.array([envs.single_action_space.sample() for _ in range(args.num_envs)])
        else: 
            pmfs = q_network.apply(q_state.params, obs)
            q_vals = (pmfs * q_state.atoms).sum(axis=-1)
            actions = q_vals.argmax(axis=-1)
            actions = jax.device_get(actions) #from gpu to host device 
        next_obs, rewards, terminations, truncations,infos = envs.step(actions)
        if "final_info" in infos: 
            for info in infos['final_info']: 
                if info and "episode" in info: 
                    print(f"Global step={step}, episodic_return={info['episode']['r']}")
        #Adding to replay buffer 
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs 
        if step > args.learning_starts and step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)
            loss, old_val, q_state = update(
                q_state,
                data.observations.numpy(),
                data.actions.numpy(),
                data.next_observations.numpy(),
                data.rewards.numpy(),
                data.dones.numpy(),
            )
            if step % 100 == 0: 
                print("SPS", int(step / (time.time() - start_time)))
            #Update target network 
            if step % args.target_network_frequency == 0:
                q_state = q_state.replace(target_params=optax.incremental_update(q_state.params, q_state.target_params, 1))
    envs.close()