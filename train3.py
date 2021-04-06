#!/usr/bin/env python
# encoding: utf-8

from NR_IES.envs.NR_IES_env import NR_IES_v0
from ray.tune.registry import register_env
import gym
import os
import ray
import ray.rllib.agents.ppo as ppo # here we can use DDPG
import shutil
def main ():
	
    # initiate directory and save checkpoints
    chkpt_root = "v1_4_6_6_9_ppo/NR_IES"
    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)
    # initialing directory to log the results
    ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)
    # starting Ray -- add `local_mode=True` here for debugging
    ray.init(ignore_reinit_error=True)
    # custom environment registration
    select_env = "NR_IES-v0"
    register_env(select_env, lambda config: NR_IES_v0())
    # create agent and environment configuration

    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"

    agent = ppo.PPOTrainer(config, env=select_env)
    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
    n_iter = 450

    # train a policy with RLlib using PPO
    for n in range(n_iter):
        result = agent.train()
        chkpt_file = agent.save(chkpt_root)
        print(status.format(
                n + 1,
                result["episode_reward_min"],
                result["episode_reward_mean"],
                result["episode_reward_max"],
                result["episode_len_mean"],
                chkpt_file
                ))


    # test the trained policy
    policy = agent.get_policy()
    model = policy.model
    print(model.base_model.summary())

    # use the trained policy in a rollout
    agent.restore(chkpt_file)
    env = gym.make(select_env)

    state = env.reset()
    sum_reward = 0
    n_step = 20

    for step in range(n_step):
        action = agent.compute_action(state)
        state, reward, done, info = env.step(action)
        sum_reward += reward

        if done == 1:
            # report at the end of each episode
            print("cumulative reward", sum_reward)
            state = env.reset()
            sum_reward = 0


if __name__ == "__main__":
    main()
