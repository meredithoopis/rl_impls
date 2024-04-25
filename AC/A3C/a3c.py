#Because tf does not support multiprocessing, gotta try with jax(best is to implement with torch, which supports threading)
import gym 
import os 
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'
import jax
import jax.numpy as jnp 
from jax import grad, vmap
import jax.random as random 
from flax import linen as nn 
from flax.training import train_state 
from flax.training.common_utils import get_metrics, onehot, shard
import optax 
import multiprocessing as mp 
import numpy as np 


class ActorCritic(nn.Module): 
    def setup(self): 
        self.shared = nn.Dense(128)
        self.actor = nn.Dense(2)
        self.critic = nn.Dense(1)
    
    def __call__(self,x): 
        shared = nn.relu(self.shared(x))
        return self.actor(shared), self.critic(shared)

def create_model(): 
    return ActorCritic()

@jax.jit    
def train_step(state,batch): 
    def loss_fn(params): 
        action_logits, values = state.apply_fn({'params': params}, batch['obs'])
        actions = jax.nn.log_softmax(action_logits, axis=-1)
        values = values.squeeze()
        critic_loss = jnp.mean((batch['returns'] - values) ** 2)
        actor_loss = -jnp.mean(actions * (batch['returns'] - values))
        return critic_loss + actor_loss 
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    return state.apply_gradients(grads=grad), loss 
    
def worker_process(worker_id,  params_queue, grads_queue): 
    env = gym.make('CartPole-v1')
    model = create_model()
    tx = optax.adam(1e-3)
    state = train_state.TrainState.create(apply_fn= model.apply, params = params_queue.get(), tx = tx )

    episode_count = 0
    while True: 
        obs = env.reset()
        done = False 
        rewards = []
        observations = []
        total_reward = 0 
        while not done: 
            action_logits, _ = model.apply({'params': state.params}, np.array([obs]))
            action = jnp.argmax(jax.nn.softmax(action_logits))
            obs, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            observations.append(obs)
            total_reward += reward
            if done: 
                R = 0 if done else model.apply({'params': state.params}, np.array([obs]))[1]
                returns = []
                for r in reversed(rewards):
                    R = r + 0.99 * R  
                    returns.insert(0,R)
                state, _ = train_step(state, {'obs': np.array(observations), 'returns': np.array(returns)})
                grads_queue.put(state.grads)
                state = state.replace(params=params_queue.get()) 
                episode_count += 1 
                if episode_count % episode_count == 0: 
                    print(f"Worker {worker_id}, Episode {episode_count}, Reward: {total_reward}")



if __name__ == "__main__": 
    mp.set_start_method('spawn')
    model = create_model()
    params = model.init(jax.random.PRNGKey(0), jnp.ones([4]))
    params_queue = mp.Queue()
    grads_queue = mp.Queue()
    workers = [mp.Process(target=worker_process,args = (i, params_queue, grads_queue)) for i in range(mp.cpu_count())]
    for worker in workers: 
        worker.start()

    for _ in range(1800): 
        grads = [grads_queue.get() for _ in range(len(workers))]
        avg_grads = jax.tree_multimap(lambda *xs: sum(xs) / len(xs), *grads)
        params = optax.apply_updates(params, avg_grads)
        for _ in range(len(workers)):
            params_queue.put(params)
    
    for worker in workers: 
        worker.join()

