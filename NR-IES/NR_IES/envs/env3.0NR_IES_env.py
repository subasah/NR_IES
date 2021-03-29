# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 12:04:11 2021

@author: ponki
"""
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


class NR_IES_v0(gym.Env):
    metadata = {
        'render.modes': ['human'],    
    }
    def __init__(self):
        self.total_t_min = 5000
        self.total_t_max = 700000000000000

        self.total_re_min = 10 
        self.total_re_max= 100000000000000

        self.e_price_min = 0.00010
        self.e_price_max = 0.00020000000

        self.h_price_min = 2
        self.h_price_max = 4000000

        self.bes_min = 200
        self.bes_max = 100000000000

        self.tes_min = 100      
        self.tes_max = 50000000000

        self.hss_min = 10        
        self.hss_max = 100000000

        self.power_grid_min = 100        
        self.power_grid_max = 500000000000000

        self.hpr_HTSE_min = 0        
        self.hpr_HTSE_max = 400000000

        self.htr_PEM_min = 0       
        self.htr_PEM_max = 200000000

        self.h_distributed_min = 0        
        self.h_distributed_max = 500000000
        
        self.episodlenght = 0
        
        self.low_state = np.array(
            [self.total_t_min, self.total_re_min, self.e_price_min, self.h_price_min, 
             self.bes_min, self.tes_min,self.hss_min, self.power_grid_min,
             self.hpr_HTSE_min,self.htr_PEM_min, self.h_distributed_min], dtype=np.float32)
        
        self.high_state = np.array(
            [self.total_t_max, self.total_re_max, self.e_price_max, self.h_price_max,
             self.bes_max, self.tes_max,self.hss_max, self.power_grid_max,
             self.hpr_HTSE_max, self.htr_PEM_max, self.h_distributed_max], dtype=np.float32)

        #actions      
        #1 Thermal power to HTSE  (10 watt- 50 watt)
        #2 Electric power to HTSE (Can we just assume it is proportional to A_1^tℎ?) (same*800)
        #3 Thermal power to thermal energy storage (50kw-800kw) ?
        #4 Electric power to battery energy storage (1kwh-100kwh) ?
        #5 Electric power to PEM (5kwh- 50kwh)
        #6 Produced hydrogen to hydrogen energy storage 1-200kg) ?

        self.low_action = np.array(
            [8000, 50000, 1000, 5000, 1 ], dtype=np.float32)
        
        self.high_action = np.array(
            [40000, 800000, 100000, 500000, 200], dtype=np.float32)

        # self.low_action = np.array(
        #     [800, 500, 100, 500, 1 ], dtype=np.float32)
        
        # self.high_action = np.array(
        #     [4000, 8000, 1000, 50000, 20], dtype=np.float32)

        self.action_space = spaces.Box(
            low=self.low_action,
            high=self.high_action,
            shape=(5,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            shape=(11,),
            dtype=np.float32
        )
        self.seed()

        # ???
        self.epen = 0
        self.tpen = 0
        self.hpen = 0
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def step(self, action):

        # assert self.action_space.contains(action)

        reward = 0
        
        total_t = self.state[0]
        total_re = self.state[1]
        e_price = self.state[2]
        h_price = self.state[3]
        bes = self.state[4]
        tes = self.state[5]
        hss = self.state[6]
        power_grid = self.state[7]
        hpr_HTSE = self.state[8]
        htr_PEM = self.state[9]
        h_distributed = self.state[10]
                
        t_htse = action[0]        
        t_stg = action[1]  
        e_stg = action[2] 
        e_pem = action[3] 
        h_stg = action[4]

        cost = 10000

        #action1  Thermal energy to HTSE
        total_t -= t_htse
        
        #action2   Thermal energy to thermal energy storage     
        if tes >= 50000:
            total_t += t_stg
        else:
            total_t -= t_stg
            tes += t_stg
            
        total_e = total_t + total_re

        #action1 Electric energy to HTSE ??
        t = t_htse 
        e = t_htse
        h = (t+e)/33333
        hpr_HTSE += h
        total_e -= e
            
            
        #action3 Electric power to battery energy storag        
        if bes >= 1000:
            total_e += e_stg
        else:
            total_e -= e_stg
            bes += e_stg
        
        #action4  Electric power to PEM 
        total_e -= e_pem
        htr_PEM = e_pem/33333
        
        total_h = hpr_HTSE + htr_PEM
        #action5  Produced hydrogen to hydrogen energy storage
        if hss >= 200:
            total_h += h_stg
        else:
            total_h -= h_stg
            hss += h_stg
        
        # Reduce episode length by 1 second
        self.episodlenght +=1 
        # Calculate reward
        profit = ((total_e*e_price)+ (total_h*h_price)) - cost
        if profit >= 10000000:
            print("asd")
            reward = 1 
        else: 
            reward = -1 

        #change reward if storage is deviated 
        if bes <= 0.2*100 and bes >= 0.8*100:
            reward = -1
            self.epen += 1
            print("BES")                        
        if tes <= 0.25*500 and tes >= 0.90*500:
            reward = -1
            self.tpen += 1
            print("TES")
        if hss <= 0.3*200 and hss >= 0.90*200:
            reward =-1
            self.hpen += 1
            print("HPES")

        
         # Check if episode is done
        if self.episodlenght >= 99: 
            done = True

        else:
            done = False
            
        observation = self.observation_space.sample()
        total_t = observation[0]
        total_re = observation[1]

        e_price = observation[2]
        h_price = observation[3]   
        
        # Set placeholder for info    
        info = {}        
        # Return step information
        # 39240.209981262204 53871.25537939453 203.30184477340697 1100.0 12.999734070766495 0.9732631115373653 2.5188
        self.state = np.array([total_t, total_re, e_price, h_price, bes, tes, hss, power_grid, hpr_HTSE, htr_PEM, h_distributed])
        # self.state = np.array([519871970.0, 21664430.0, 0.00018702277, 2.1713088, 69221.33, 10828.11, 83.45252, 286424830.0, 385.82718, 78.96604, 133.4819])
        # self.state = [510000010, 11000010, 0.00011702277, 2.0791678, 2100.287, 2054.404, 13.59362, 1100.0, 3.4513, 2.14204, 2.5188]

        try:
            assert self.observation_space.contains(self.state)
        except AssertionError:
            print("INVALID STATE", self.state)
        return self.state, reward, done, info
    
    def reset(self):
        # Reset shower temperature
        # self.state = [600000000,50000000,0.15,2,50000,20000,25,50000,10,5,100]
        # self.state = [572980350.0, 37890804.0, 0.00015191699, 2.0791678, 16692.ls287, 10548.404, 91.59362, 72073250.0, 364.4513, 85.14204, 483.5188]
        self.state = [510000010, 11000010, 0.00015191699, 2.0791678, 2100.287, 2054.404, 13.59362, 1100.0, 3.4513, 2.14204, 2.5188]
        
        # Reset shower time
        self.epen = 0
        self.tpen = 0
        self.hpen = 0

        self.episodlenght = 0
        return self.state

env = NR_IES_v0()
episodes = 20
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    store = 0
    while not done:
        #env.render()
        action = env.action_space.sample()        
        n_state, reward, done, info = env.step(action)
        print(env.epen, env.tpen, env.hpen)
        # print(len(n_state), n_state[])
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))