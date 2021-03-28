#!/usr/bin/env python
# encoding: utf-8

import gym
# import gym_example
from NR_IES.envs.NR_IES_env import NR_IES_v0
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

def run_one_episode(env, verbose=False):
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

    #first, create the custom environment and run it for one episode
    env = gym.make("NR_IES-v0")

    # print(env.action_space.sample())
    # print(env.observation_space.sample())
    # print(type(np.array([12.5,  5. ,  2. ,  2.5,  2. ,  2. ])))

    #print(type(env.observation_space.sample()))
    #ValueError: ('Observation ({}) outside given space ({})!', array([12.5,  5. ,  2. ,  2.5,  2. ,  2. ]), Box(-10.0, 20.0, (6,), float32))
    
    # print(env.observation_space.contains(np.array([6.0e+08, 5.0e+06, 1.5e-01, 2.0e+00, 5.0e+03, 2.0e+06, 2.5e+01, 5.0e+03, 1.0e+02, 5.0e+01, 2.0e+02])))
    # print(env.observation_space.contains(np.array([600000000.0,5000000.0,0.15,2.0,5000.0,2000000.0,25.0,5000.0,100.0,50.0,200.0])))
    # exit()
    ob = env.observation_space.sample()
    print(ob[0], ob[1], ob[2], ob[3], ob[4], ob[5], ob[6], ob[7], ob[8], ob[9], ob[10])
    
    
    print(env.observation_space.contains(np.array([519871970.0, 21664430.0, 0.00018702277, 2.1713088, 69221.33, 10828.11, 83.45252, 286424830.0, 385.82718, 78.96604, 133.4819])))
    exit()

    # array()
    # observation_space.contains(observation)

    # observation = self.observation_space.sample()
    # total_T =observation[0]
    # total_RE =observation[1]
    
    # e_price = observation[2]
    # h_price = observation[3]  
    # exit()
    # total_T = []
    # e_price = []
    # h_price = []
    # total_re = []
    # bes = []
    # for i in range(100):
        # observation = env.observation_space.sample()
        # total_T.append(observation[0])
        # e_price.append(observation[2])
        # h_price.append(observation[3])
        # total_re.append(observation[1])
        # bes.append(observation[4])
        # tes_stat.append(observation[5])
        # hyd_stat.append(observation[6])
        # epg.append(observation[7])
        # hprate.append(observation[8])
        # hpratep.append(observation[9])
        # hsm.append(observation[10])

    # plt.subplot(5,1,1)
    # plt.plot(e_price, 'b') 
    # plt.ylabel('e_price')
    # plt.subplot(5,1,2)
    # plt.plot(total_T,'g')  
    # plt.ylabel('total_T')
    # plt.subplot(5,1,3)
    # plt.plot(total_re,'r')
    # plt.ylabel('total_RE')
    # plt.subplot(5,1,4)
    # plt.plot(h_price,'y')
    # plt.ylabel('h_price')
    # plt.subplot(5,1,5)
    # plt.plot(bes,'m')
    # plt.ylabel('BES') 
    # plt.xlabel('episode')
    # plt.show()


    tes_stat = []
    hyd_stat = []
    epg = []
    hprate = []
    hpratep = []
    hsm = []

    for i in range(100):
        observation = env.observation_space.sample()
        tes_stat.append(observation[5])
        hyd_stat.append(observation[6])
        epg.append(observation[7])
        hprate.append(observation[8])
        hpratep.append(observation[9])
        hsm.append(observation[10])

    plt.subplot(6,1,1)
    plt.plot(hyd_stat, 'c') 
    plt.ylabel('hyd_stat')
    plt.subplot(6,1,2)
    plt.plot(tes_stat,'k')  
    plt.ylabel('tes_stat')
    plt.subplot(6,1,3)
    plt.plot(hprate,'r')
    plt.ylabel('hprate')
    plt.subplot(6,1,4)
    plt.plot(epg,'g')
    plt.ylabel('epg')
    plt.subplot(6,1,5)
    plt.plot(hpratep,'m')
    plt.ylabel('hpratep') 

    plt.subplot(6,1,6)
    plt.plot(hpratep,'y')
    plt.ylabel('hsm') 

    plt.xlabel('episode')
    plt.show()


    # print(type(env.observation_space.sample()))
    #ValueError: ('Observation ({}) outside given space ({})!', array([12.5,  5. ,  2. ,  2.5,  2. ,  2. ]), Box(-10.0, 20.0, (6,), float32))
    # print(env.observation_space.contains(np.array([12.5,  5. ,  2. ,  2.5,  2. ,  2. ])))


    #this is what we need here
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