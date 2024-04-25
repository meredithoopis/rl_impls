#Ref: https://huggingface.co/blog/deep-rl-a2c: For the official formula 
import gym 
import numpy as np 
from actor_critic import Agent 
from gym import wrappers 
from utils import plot_learning_curve



if __name__ == "__main__": 
    env = gym.make('CartPole-v1')
    agent = Agent(alpha = 1e-5, n_actions=env.action_space.n)
    n_games = 1800 
    #Record 
    #env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    file_name = 'cartpole.png'
    figure_file = 'plots/' + file_name 
    best_score = env.reward_range[0]
    score_history = []
    for i in range(n_games): 
        observation, _  = env.reset()
        done = False 
        score = 0 
        while not done: 
            action = agent.choose_action(observation)
            observation_, reward, done,truncated, info = env.step(action)
            #observation_, _ = observation_ 
            score += reward 
            agent.learn(observation, reward, observation_,done)
            observation = observation_
            
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score: 
            best_score = avg_score
        print('episode', i , 'score %.1f' % score, 'avg_score %.1f' % avg_score)
    
    x = [i + 1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)