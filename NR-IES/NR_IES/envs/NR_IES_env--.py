import gym
from gym import Env
from gym import spaces
from gym.spaces import Discrete, Box
from gym.utils import seeding
import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv("da_hrl_lmps.csv")
df_wind = pd.read_csv("wind_gen.csv")

eprice = df['total_lmp_da'].head(120).values.tolist()
wind_generation = df_wind['wind_generation_mw'].head(120).values.tolist()

sample = 120
x = np.arange(sample)

# plt.subplot(2,1,1)
# plt.plot(x, eprice,'r')
# plt.subplot(2,1,2)
# plt.plot(x, wind_generation,'r')
# plt.show()
# [3.35907642e+03, 6.62790000e+01, 1.64618950e+01, -1.00000012e-01]
class NR_IES_v0(gym.Env):
    metadata = {
        'render.modes': ['human'],    
    }
    def __init__(self):

        self.total_T_min = 599 
        self.total_T_max = 6010

        self.total_RE_min = 1
        self.total_RE_max= 1500


        self.e_price_min = 1
        self.e_price_max = 200

        self.bes_min = -1  
        self.bes_max = 5
        
        # [4.01671460e+03 7.51190000e+01 1.64700000e+01 1.02272789e+00]
        self.reward = 0
        
        self.low_state = np.array(
            [self.total_T_min, self.total_RE_min, self.e_price_min, self.bes_min], dtype=np.float32
            )
        
        self.high_state = np.array(
            [self.total_T_max, self.total_RE_max, self.e_price_max, self.bes_max], dtype=np.float32
            )
                
        self.action_space = spaces.Box(
            low=-0.3,
            # low = 0,
            high = 0.25,
            shape = (1,),
            dtype = np.float32
        )
        self.observation_space = spaces.Box(
            low = self.low_state,
            high = self.high_state,
            dtype = np.float32
        )
        self.seed()       
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def step(self, action): 

        assert self.action_space.contains(action)

        n_energy = self.state[0]         
        r_energy = self.state[1] 
        e_price =self.state[2]

        BES_SOC = self.state[3] 

        BES_update = action[0]   
                 
        # Reduce episode length by 1 second
        self.episodlenght += 1 

        # Calculate CHANGES
        TOTAL_T = n_energy + r_energy
        BES_SOC += BES_update
        TOTAL = TOTAL_T - BES_SOC*300

         # Calculate reward
        profit = ((TOTAL*e_price))
        
        if profit > 11000:
            self.reward += 6
        else: 
            self.reward += -5 
         
         # Check if episode is done                
        if self.episodlenght >= 119 or BES_SOC < 0.10 or BES_SOC > 0.98: 
            done = True
            plt.show()
        else:
            done = False
            
        os = self.observation_space.sample()
        thermal_energy = os[0]
        n_energy = thermal_energy         
        r_energy = wind_generation[self.episodlenght]
        e_price = eprice[self.episodlenght]        
        # Set placeholder for info    
        info = {}        
        # Return step information
        # statef = []
        # statef.append(Allvariable(i,f,md))
        self.state = np.array([n_energy, r_energy, e_price, BES_SOC])

        try:
            assert self.observation_space.contains(self.state)
        except AssertionError:
            print("INVALID STATE", self.state)
            
        return self.state, self.reward, done, info
    
    def reset(self):
        # Reset shower temperature
        self.state = [self.observation_space.sample()[0], wind_generation[0], eprice[0], 0.9]
        # Reset shower time
        self.episodlenght = 0
        return self.state
    
env = NR_IES_v0()
episodes = 1000
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0 
    while not done:
        #env.render()
        action = env.action_space.sample()        
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))