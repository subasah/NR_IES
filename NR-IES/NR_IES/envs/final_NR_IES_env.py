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
        # self.total_t_min = 500000000
        # self.total_t_max = 7000000000
        # self.total_re_min = 1000000 
        # self.total_re_max= 1000000000
        # self.e_price_min = 0.10 
        # self.e_price_max = 0.20
        # self.h_price_min = 1.5
        # self.h_price_max = 4
        # self.bes_min = 2000
        # self.bes_max = 1000000
        # self.tes_min = 10 
        # self.tes_max = 50000000
        # self.hss_min = 10        
        # self.hss_max = 1000
        # self.power_grid_min = 1000        
        # self.power_grid_max = 500000000
        # self.hpr_HTSE_min = 1        
        # self.hpr_HTSE_max = 4000
        # self.htr_PEM_min = 1        
        # self.htr_PEM_max = 2000
        # self.h_distributed_min = 1        
        # self.h_distributed_max = 6000

        self.total_t_min = 0.0
        self.total_t_max = 7000000000000000000
 
        self.total_re_min = 0.0 
        self.total_re_max= 10000000000000000000000

        self.e_price_min = 0.0
        self.e_price_max = 0.90000000000000000000

        self.h_price_min = 0.0
        self.h_price_max = 400000000000000000

        self.bes_min = 0.0
        self.bes_max = 10000000000000000

        self.tes_min = 0.0      
        self.tes_max = 500000000000000000

        self.hss_min = 0.0        
        self.hss_max = 100000000000000000

        self.power_grid_min = 0.0        
        self.power_grid_max = 500000000000000000

        self.hpr_HTSE_min = 0.0        
        self.hpr_HTSE_max = 4000000000000000000

        self.htr_PEM_min = 0.0       
        self.htr_PEM_max = 20000000000000000000000

        self.h_distributed_min = 0.0        
        self.h_distributed_max = 5000000000000000

        self.reward = 0 

        self.episodlenght = 0

        self.eprofit_data = []
        self.hprofit_data = []
        self.tprofit = []
        
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
        # self.low_action = np.array(
        #     [10, 50000, 1000, 5000, 1 ], dtype=np.float32)
        
        # self.high_action = np.array(
        #     [500, 800000, 100000, 50000, 200], dtype=np.float32)

        self.low_action = np.array(
            [800, 5000, 100, 500, 1], dtype=np.float32)
        
        self.high_action = np.array(
            [4000, 80000, 10000, 50000, 50], dtype=np.float32)

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

        self.epen = 0
        self.tpen = 0
        self.hpen = 0
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action): 
        total_T = self.state[0]
        total_RE = self.state[1]
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
        e_pem= action[3] 
        h_stg= action[4] 
        
        #action1  Thermal power to HTSE
        total_T -= t_htse
        
        #action2   Thermal power to thermal energy storage     
        if tes>=500000:
            total_T += t_stg
        else:
            total_T -=t_stg
            tes += t_stg
            
        elec = total_T + total_RE
        #action1 Electric power to HTSE 
        t = t_htse 
        e = 800*t_htse
        h = (t+e)/33333
        hpr_HTSE += h
        elec -= e

        #action3 Electric power to battery energy storag        
        if bes >= 10000:
            elec += e_stg
        else:
            elec -= e_stg
            bes += e_stg
        
        #action4  Electric power to PEM 
        elec -= e_pem
        htr_PEM = e_pem/33333
        
        total_h = hpr_HTSE + htr_PEM
        #action5  Produced hydrogen to hydrogen energy storage
        if hss >= 200:
            total_h += h_stg
        else:
            total_h -= h_stg
            hss += h_stg
        
        cost = 10

        # Calculate reward
        profit_e = elec*e_price 
        profit_h =  total_h*h_price
        profit = profit_e + profit_h - cost
        # elec =0
        # total_h=0
        
        
        if profit >= 800000000:
        # if profit >= 8000000000000:
            self.reward += 5.6
            print("PROFIT")
        else: 
            self.reward -= 3

        #change reward if storage is deviated 
        if bes<= 0.2*1000000 or bes>= 0.8*1000000:
            self.reward -= 1.5
            print("BES")
            # self.epen += 1                        
        if tes<= 0.25*50000000 or tes >= 0.90*50000000:
            self.reward -= 1.8
            print("TES")
            # self.tpen += 1
        if hss<= 0.3*1000 or hss >= 0.90*1000:
            self.reward -= 2.0
            # print(self.reward)
            print("HSS")
            # self.hpen += 1
        
         # Reduce episode length by 1 second
        self.episodlenght += 1 

        # Check if episode is done
        if self.episodlenght >= 2: 
            done = True
            # self.eprofit_data.append(profit_e)
            # self.hprofit_data.append(profit_h)
            # self.tprofit.append(profit)

        else:
            done = False
            # self.eprofit_data.append(profit_e)
            # self.hprofit_data.append(profit_h)
            # self.tprofit.append(profit)
            
        observation = self.observation_space.sample()
        total_T = observation[0]
        total_RE = observation[1]
        e_price = observation[2]
        h_price = observation[3]   
        
        # Set placeholder for info    
        info = {}        

        self.state = np.array([total_T, total_RE, e_price, h_price, bes, tes, hss, power_grid, hpr_HTSE, htr_PEM, h_distributed])

        try:
            assert self.observation_space.contains(self.state)
        except AssertionError:
            print("INVALID STATE", self.state)

        return self.state, self.reward, done, info
    
    def reset(self):
        # Reset shower temperature
        # self.state = [510000010, 11000010, 0.00015191699, 2.0791678, 2100.287, 2054.404, 13.59362, 1100.0, 3.4513, 2.14204, 2.5188]
        self.state = [510000010, 11000010, 0.15191699, 2.0791678, 2100.287, 2054.404, 13.59362, 1100.0, 3.4513, 2.14204, 22.5188]
        # [5.1000001e+08, 1.1000010e+07, 1.5191699e-04, 2.0791678e+00,
        #        2.1002870e+03, 2.0544040e+03, 1.3593620e+01, 1.1000000e+03,
        #        3.4513000e+00, 2.1420400e+00, 2.5188000e+00])
        # Reset shower time
        self.epen=0
        self.tpen=0
        self.hpen=0

        self.reward = 0 
        self.episodlenght = 0

        return self.state

    # def close(self):
    #     plt.subplot(3,1,1)
    #     plt.plot(self.eprofit_data, 'c') 
    #     plt.ylabel('e_profit')
    #     plt.subplot(3,1,2)
    #     plt.plot(self.hprofitdata,'k')  
    #     plt.ylabel('h_profit')
    #     plt.subplot(3,1,3)
    #     plt.plot(self.profit,'r')
    #     plt.ylabel('profit')
    #     plt.show()


env = NR_IES_v0()
episodes = 1
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    while not done:
        #env.render()
        action = env.action_space.sample()        
        n_state, reward, done, info = env.step(action)
        # print("reward", reward)
        # print(env.epen, env.tpen, env.hpen)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))