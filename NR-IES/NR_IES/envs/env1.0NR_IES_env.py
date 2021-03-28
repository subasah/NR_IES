import gym
from gym import Env
from gym import spaces
from gym.spaces import Discrete, Box
from gym.utils import seeding
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import numpy as np

# Fs = 100
# f = 5
# sample = 100
# x = np.arange(sample)
# xx = np.arange(100)
# y = (2*np.sin(2 * np.pi * f * x / Fs)) +2
# z = (2.5*np.sin(2 * np.pi * f * 2*x / Fs)) +2.5
# randnums= np.random.randint(-3,4,100)
# v= (2*z-y)+2
# w = 2*(z+y)+3.5

# print(type(w), type(y), type(v), type(z))
# exit()

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
    
class NR_IES_v0(gym.Env):
    metadata = {
        'render.modes': ['human'],    
    }
    def __init__(self):

        self.h_minprice = 0.0
        self.h_maxprice = 10.0

        #trained with these 
        self.e_minprice = 1.0
        self.e_maxprice = 20.0
        #rollout  
        # self.e_minprice = -10
        # self.e_maxprice = 20.0
        #
        self.h_mindemand = 0.0
        self.h_maxdemand = 4.5

        self.e_mindemand = 0.0
        self.e_maxdemand = 5.5

        self.h_minprodrate = 0.0
        # self.h_maxprodrate = 5.0
        self.h_maxprodrate = 200.0

        self.e_minprodrate = 0.0
        # self.e_maxprodrate = 6.0
        self.e_maxprodrate = 150.0

        self.episodlenght = 0.0
        # self.e_prodrate = []
        # self.h_prodrate = []
        # self.profit = []

        self.low_state = np.array(
            [self.h_minprice, self.e_minprice, self.h_mindemand, self.e_mindemand, self.h_minprodrate, self.e_minprodrate], dtype=np.float32)
        
        self.high_state = np.array(
            [self.h_maxprice, self.e_maxprice, self.h_maxdemand, self.e_maxdemand, self.h_maxprodrate, self.e_maxprodrate], dtype=np.float32)
        
        self.action_space = spaces.Box(
            low=0.0,
            high=4.0,
            shape=(2,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )
        self.seed()
        self.reset()       
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action): 
        assert self.action_space.contains(action)

        h_price = self.state[0]            
        e_price = self.state[1] 
        h_demand =self.state[2] 
        e_demand = self.state[3] 
        h_prodrate = self.state[4]        
        e_prodrate = self.state[5]
        # h_prodrate = 0
        # e_prodrate = 0          
        h_prodrate += action[0]        
        e_prodrate += action[1]  

        # Reduce episode length by 1 second
        self.episodlenght +=1 


        # Calculate reward
        profit = (((e_prodrate*e_price)+ (h_prodrate*h_price))-((e_demand*e_price)+ (h_demand*h_price)))
        # profit = 70
        if profit >=50 and profit <=150: 
            reward = 1 
        else: 
            reward = -1 

        if self.episodlenght >= 99: 
            done = True
        else:
            done = False
            # self.e_prodrate.append(e_prodrate)
            # self.h_prodrate.append(h_prodrate)
            # self.profit.append(profit)

        #sample from observation_space and then do this
        observation = self.observation_space.sample()
        h_price = observation[0]           
        e_price = observation[1]
        h_demand = observation[2]
        e_demand = observation[3]

        # self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        # return np.array(self.state)
        
        # h_price = w[self.episodlenght]           
        # e_price = v[self.episodlenght]
        # h_demand =y[self.episodlenght] 
        # e_demand =z[self.episodlenght]

        # Set placeholder for info    
        info = {}        
        # Return step information
        # statef = []
        # statef.append(Allvariable(i,f,md))
        self.state = np.array([h_price, e_price, h_demand, e_demand, h_prodrate, e_prodrate])

        try:
            assert self.observation_space.contains(self.state)
        except AssertionError:
            print("INVALID STATE", self.state)

        return self.state, reward, done, info
    
    def reset(self):
        # Reset shower temperature
        # self.state = [6, 20, 30, 40, 50, 45]
        self.state = [3, 3, 3, 3, 3, 3]
        # Reset shower time
        self.episodlenght = 0
        return self.state


# env = CostomIESEnv()
# episodes = 2
# for episode in range(1, episodes+1):
#     state = env.reset()
#     done = False
#     score = 0 
#     while not done:
#         #env.render()
#         action = env.action_space.sample()        
#         n_state, reward, done, info = env.step(action)
#         score+=reward
#     print('Episode:{} Score:{}'.format(episode, score))
