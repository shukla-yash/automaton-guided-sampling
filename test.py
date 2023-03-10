import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym

from PPO import PPO

import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper

#################################### Testing ###################################
def test():
    print("============================================================================================")

    ################## hyperparameters ##################

    # env_name = "CartPole-v1"
    # has_continuous_action_space = False
    # max_ep_len = 400
    # action_std = None

    # env_name = "LunarLander-v2"
    # has_continuous_action_space = False
    # max_ep_len = 300
    # action_std = None

    # env_name = "BipedalWalker-v2"
    # has_continuous_action_space = True
    # max_ep_len = 1500           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving

    # env_name = "RoboschoolWalker2d-v1"
    env_name = "MiniGrid-NineRoomsHardKey-v0"
    has_continuous_action_space = False 
    max_ep_len = 500           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving

    render = False              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    total_test_episodes = 1000    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################

    env = gym.make(env_name, max_steps = max_ep_len, render_mode="human")
    # env = RGBImgObsWrapper(env)
    env = FullyObsWrapper(env)
    # state space dimension
    state_dim = env.observation_space.shape

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n-1

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0
    done_arr = []
    reward_arr = []

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state, info = env.reset()

        for t in range(1, max_ep_len+1):
            state['image'] = np.swapaxes(state['image'],0,2)
            state['image'] = np.expand_dims(state['image'], axis=0)
            action = ppo_agent.select_action(state['image'])    
            state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            # time.sleep(1)

            if terminated or truncated:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))
    print("Success rate : ", np.mean(done_arr))

    print("============================================================================================")


if __name__ == '__main__':

    test()
