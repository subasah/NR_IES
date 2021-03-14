#!/usr/bin/env python
# encoding: utf-8

import gym
# import gym_example
# from NR_IES.envs.NR_IES_env import NR_IES_v0
import NR_IES
import numpy as np

# for plottings
from gym import Env
from gym import spaces
from gym.spaces import Discrete, Box
from gym.utils import seeding
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import numpy as np

def run_one_episode (env, verbose=False):
    env.reset()
    sum_reward = 0

    # for i in range(env.MAX_STEPS):
    for i in range(1):
        action = env.action_space.sample()

        if verbose:
            print("action:", action)

        state, reward, done, info = env.step(action)
        sum_reward += reward

        if verbose:
            env.render()

        if done:
            if verbose:
                print("done @ step {}".format(i))

            break

    if verbose:
        print("cumulative reward", sum_reward)

    return sum_reward


def main ():
    # first, create the custom environment and run it for one episode
    env = gym.make("NR_IES-v0")
    print(env.action_space.sample())
    print(type(np.array([12.5,  5. ,  2. ,  2.5,  2. ,  2. ])))

    h_price = []
    e_price = []
    h_demand = []
    e_demand = []

    for i in range(100):
        observation = env.observation_space.sample()
        h_price.append(observation[0])
        e_price.append(observation[1])
        h_demand.append(observation[2])
        e_demand.append(observation[3])

    plt.subplot(4,1,1)
    plt.plot(e_price, 'b')  #electricity price
    plt.ylabel('e_price')
    plt.subplot(4,1,2)
    plt.plot(h_price,'g')  #hydrogen price
    plt.ylabel('h_price')
    plt.subplot(4,1,3)
    plt.plot(e_demand,'r')
    plt.ylabel('e_demand')
    plt.subplot(4,1,4)
    plt.plot(h_demand,'y') 
    plt.xlabel('episode')
    plt.ylabel('h_demand')
    plt.show()



    # h_price = w[self.episodlenght]           
    # e_price = v[self.episodlenght]
    # h_demand =y[self.episodlenght] 
    # e_demand =z[self.episodlenght]

    # plt.subplot(4,1,1)
    # plt.plot(x, v, 'b')  #electricity price
    # plt.ylabel('e_price')
    # plt.subplot(4,1,2)
    # plt.plot(x, w,'g')  #hydrogen price
    # plt.ylabel('h_price')
    # plt.subplot(4,1,3)
    # plt.plot(x, y,'r')
    # plt.ylabel('e_demand')
    # plt.subplot(4,1,4)
    # plt.plot(x, z,'y') 
    # plt.xlabel('episode')
    # plt.ylabel('h_demand')
    # plt.show()
    # plt.figure()
    # plt.scatter(xx, randnums)
    # plt.show()

    # print(type(env.observation_space.sample()))
    #ValueError: ('Observation ({}) outside given space ({})!', array([12.5,  5. ,  2. ,  2.5,  2. ,  2. ]), Box(-10.0, 20.0, (6,), float32))
    # print(env.observation_space.contains(np.array([12.5,  5. ,  2. ,  2.5,  2. ,  2. ])))
    exit()

    # NR_IES-v0
    sum_reward = run_one_episode(env, verbose=True)

    # next, calculate a baseline of rewards based on random actions
    # (no policy)
    history = []

    for _ in range(10000):
        sum_reward = run_one_episode(env, verbose=False)
        history.append(sum_reward)

    avg_sum_reward = sum(history) / len(history)
    print("\nbaseline cumulative reward: {:6.2}".format(avg_sum_reward))


if __name__ == "__main__":
    main()