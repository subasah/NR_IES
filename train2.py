#!/usr/bin/env python
# encoding: utf-8
import ray
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print
import json
import shutil

def main ():
    
    # initiate directory and save checkpoints
    chkpt_root = "stnd_env/NR_IES"
    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

    info = ray.init(ignore_reinit_error=True, log_to_driver=False)
    config = DEFAULT_CONFIG.copy()
    config['num_workers'] = 1
    config['num_sgd_iter'] = 30
    config['sgd_minibatch_size'] = 128
    config['model']['fcnet_hiddens'] = [100, 100]
    config['num_cpus_per_worker'] = 0
    agent = PPOTrainer(config, 'CartPole-v1')

    N = 200
    results = []
    episode_data = []
    episode_json = []

    for n in range(N):
        result = agent.train()
        chkpt_file = agent.save(chkpt_root)
        results.append(result)
        
        episode = {'n': n, 
                'episode_reward_min':  result['episode_reward_min'],  
                'episode_reward_mean': result['episode_reward_mean'], 
                'episode_reward_max':  result['episode_reward_max'],  
                'episode_len_mean':    result['episode_len_mean']} 
        
        episode_data.append(episode)
        episode_json.append(json.dumps(episode))
        
        print(f'{n:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}')



if __name__ == "__main__":
    main()
