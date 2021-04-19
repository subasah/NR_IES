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

# sample = 120
# x = np.arange(sample)
# plt.subplot(2,1,1)
# plt.plot(x, eprice,'r')
# plt.subplot(2,1,2)
# plt.plot(x, wind_generation,'r')
# plt.show()               
    
class NR_IES_v0(gym.Env):
    metadata = {
        'render.modes': ['human'],    
    }
    def __init__(self):

        self.total_T_min = 599 
        self.total_T_max = 601

        self.total_RE_min = 1 
        self.total_RE_max= 150

        self.e_price_min = 15 
        self.e_price_max = 20

        self.bes_min = 0     
        self.bes_max = 1
         
        self.tes_min = 0
        self.tes_max = 4000
        
        self.hes_min = 0
        self.hes_max = 5000
        
        self.re_price = []
        self.rwind = []
        self.rbes_soc = []
        self.rthermal = []
        self.rprofit = []
        self.rhes = []
        self.etes = []
        

        self.low_state = np.array(
            [self.total_T_min, self.total_RE_min, self.e_price_min, self.bes_min,  self.tes_min, self.hes_min ], dtype=np.float32
            )
        
        self.high_state = np.array(
            [self.total_T_max, self.total_RE_max, self.e_price_max, self.bes_max, self.tes_max, self.hes_max], dtype=np.float32
            )
        
        self.action_space = spaces.Tuple((Discrete(9), Box(-100, 100, (1,)), Box(0, 50, (1,)), Box(-1000, 1000, (1,))))
        
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )
        self.seed()       
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action): 
        n_energy = self.state[0]         
        r_energy = self.state[1] 
        e_price = self.state[2] 
        BES_SOC = self.state[3] 
        TES = self.state[4]
        HES = self.state[5]
        dtemp = False     
        a = 0
        
        if action[0] == 0: a = -0.8
        if action[0] == 1: a = -0.6
        if action[0] == 2: a = -0.4
        if action[0] == 3: a = -0.2
        if action[0] == 4: a = 0
        if action[0] == 5: a = 0.2
        if action[0] == 6: a = 0.4
        if action[0] == 7: a = 0.6
        if action[0] == 8: a = 0.8
        

        BES_update = a   
        tes_update = action[1]
        h_convert = action[2]
        h_storage = action[3]
                 
        # Reduce episode length by 1 second
        self.episodlenght += 1 
        
        # Calculate main CHANGES
        TOTAL_T = n_energy + r_energy
        TOTAL_T =- tes_update
        TES =+ tes_update
        TOTAL_T =- h_convert
        hydrogen =+ h_convert*16.4 #HYDROGEN PRODUCED IN KG


      #give out of bound panelty for thermal storage
        if TES > 3950 or TES < 100:
            reward = -1000
            dtemp = True  
            done = True
            TES = 0
            
        #calculate storage changes
        BES_SOC += BES_update   
        hydrogen =- h_storage
        HES =+ h_storage
       
        #calculate remained power    
        TOTAL = TOTAL_T - BES_SOC*300
        
        #give out of bound panelty for battery storage
        if BES_SOC > 0.95 or BES_SOC < 0:
            reward = -1000
            dtemp = True  
            done = True
            BES_SOC = 0

        #give out of bound panelty for hydrogen storage
        if HES > 1 or HES < 0:
            reward = -1000
            dtemp = True  
            done = True
            HES = 0
                 

         # Calculate reward
        profit = ((TOTAL*e_price)) + (hydrogen*2.40)
        
        if profit > 11000:
            reward = 30
        elif profit > 9000:
            reward = 20
        elif profit > 7000:
            reward = 10
        elif profit > 6000:
            reward = 5
        elif profit > 5000:
            reward = 3
        elif profit > 3000:
            reward = 2
        else: 
            reward = -10 
        
        # if self.episodlenght/10 == 1.0:
        #      print(self.episodlenght)
             
         # Check if episode is done     
        if dtemp == True:
            reward = -1000
            done = True
            # self.rbes_soc.append(BES_SOC)
            # self.rthermal.append(n_energy)
            # self.re_price.append(e_price)
            # self.rwind.append(r_energy)
            # self.rprofit.append(profit)
           
        elif self.episodlenght >= 119 or BES_SOC < 0.10 or BES_SOC > 0.98:
            done = True            
            # self.rbes_soc.append(BES_SOC)
            # self.rthermal.append(n_energy)
            # self.re_price.append(e_price)
            # self.rwind.append(r_energy)
            # self.rprofit.append(profit)
            # plt.show()
        else:
            done = False
            # self.rbes_soc.append(BES_SOC)
            # self.rthermal.append(n_energy)
            # self.re_price.append(e_price)
            # self.rwind.append(r_energy)
            # self.rprofit.append(profit)
           
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
        self.state = np.array([n_energy, r_energy, e_price, BES_SOC, TES, HES])
        return self.state, reward, done, info

    
    def reset(self):
        # Reset shower temperature
        self.state = [self.observation_space.sample()[0], wind_generation[0], eprice[0], 0, 0, 0]
        # Reset shower time
        self.episodlenght = 0
        return self.state
    
    # def close(self):
    #     ax1 = plt.subplot(7,1,1)
    #     plt.plot(self.rthermal,'y')
    #     plt.setp(ax1.get_xticklabels(), visible=False)
    #     plt.ylabel('rthermal')

    #     ax2 = plt.subplot(7,1,2)
    #     plt.plot(self.rbes_soc,'r')
    #     plt.setp(ax2.get_xticklabels(), visible=False)
    #     plt.yticks(np.arange(0,1,0.2))
    #     plt.ylabel('bes_soc')
     
    #     ax3 = plt.subplot(7,1,3)
    #     plt.plot(self.re_price,'b')
    #     plt.setp(ax3.get_xticklabels(), visible=False)
    #     plt.ylabel('e_price')
       
    #     ax4 = plt.subplot(7,1,4)
    #     plt.plot(self.rwind,'b')
    #     plt.setp(ax4.get_xticklabels(), visible=False)
    #     plt.ylabel('wind')
        
    #     plt.subplot(7,1,5)
    #     plt.plot(self.rprofit,'r')
    #     plt.setp(ax4.get_xticklabels(), visible=False)
    #     plt.ylabel('profit')
    
    #     plt.subplot(7,1,6)
    #     plt.plot(self.rhes,'b')
    #     plt.setp(ax4.get_xticklabels(), visible=False)
    #     plt.ylabel('hes')

    #     plt.subplot(7,1,7)
    #     plt.plot(self.rtes,'b')
    #     plt.ylabel('tes')
    #     plt.show()
    
    
# env = NR_IES_v0()
# episodes = 2
# for episode in range(1, episodes + 1):
#     state = env.reset()
#     done = False
#     score = 0
#     while not done:
#         #env.render()
#         action = env.action_space.sample()        
#         n_state, reward, done, info = env.step(action)
#         score += reward
#     print('Episode:{} Score:{}'.format(episode, score))