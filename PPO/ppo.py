#ref: https://spinningup.openai.com/en/latest/algorithms/ppo.html
import os
import random
import time
from dataclasses import dataclass
import torch 
import torch.nn as nn 
import numpy as np 
import torch.optim as optim 
import tyro 
import gymnasium as gym 
from torch.distributions.categorical import Categorical 
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv, EpisodicLifeEnv, FireResetEnv, MaxAndSkipEnv, 
    NoopResetEnv
)


@dataclass 
class Args: 
    seed: int = 1 
    torch_deterministic: bool=True
    env_id: str = "BreakoutNoFrameskip-v4"
    cuda: bool=True 
    total_timesteps: int = 5000000
    learning_rate : float = 2.5e-4 
    num_envs: int = 8 #number of parallel game envs 
    #buffer_size: int = int(1e-6)
    num_steps: int = 128 #Number of steps to run in each environment per policy rollout
    anneal_lr : bool=True #lr annealing for policy and value network 
    gamma: float = 0.99 #Discount factor 
    gae_lambda: float = 0.95 #general advantage estimation 
    num_minibatches: int =4 
    update_epochs: int = 4 
    capture_video: bool = False 
    norm_adv: bool = True #Normalize the advantage function 
    clip_coef: float = 0.1 #Clipping coef 
    clip_vloss: bool = True #use a clipped loss for the value function 
    ent_coef: float = 0.01 #coef of the entropy 
    vf_coef: float = 0.5 #coef of the value function 
    #tau: float = 0.05  #Target network update rate 
    #target_network_frequency: int = 500 #Timesteps it takes to update the target network 
    batch_size: int = 0
    #exploration_noise : float = 0.1 
    #start_e: float= 1 #Epsilon: Exploration should be large in the beginning 
    #end_e: float = 0.05 #Ending epsilon: Exploration rate 
    #exploration_fraction:  float = 0.5
    max_grad_norm: float = 0.5 #max norm for gradient clipping 
    target_kl : float = None #Target KL divergence threshold 
    #learning_starts: int = 25e3 #Learning starts each 10k steps 
    #policy_frequency: int = 2 
    #noise_clip: float = 0.5 #Noise clip param of the target policy smoothing regularization 
    minibatch_size: int = 0 
    num_iterations: int = 0 #Computed in runtime 



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

def layer_init(layer, std = np.sqrt(2), bias_const = 0.0): 
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer 

class Agent(nn.Module): 
    def __init__(self, envs): 
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4,32,8,stride=4)), 
            nn.ReLU(), 
            layer_init(nn.Conv2d(32,64,4,stride=2)), 
            nn.ReLU(), 
            layer_init(nn.Conv2d(64,64,3,stride=1)), 
            nn.ReLU(), 
            nn.Flatten(), 
            layer_init(nn.Linear(64 * 7 * 7, 512)), 
            nn.ReLU(), 

        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std = 0.01)
        self.critic = layer_init(nn.Linear(512,1), std=1)

    def get_value(self,x): 
        return self.critic(self.network(x/255.0)) #return value for actions 
    
    def get_action_and_value(self,x,action=None): 
        hidden = self.network(x/255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits = logits)
        if action is None: 
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)  #get action and according value 
    

if __name__ == "__main__": 
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size 
    run_name = f"{args.env_id}_running"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, i,args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate 
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # action
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        print("SPS:", int(global_step / (time.time() - start)))

    envs.close()