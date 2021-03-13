#!/usr/bin/env python
# encoding: utf-8

import gym
# import gym_example
# from NR_IES.envs.NR_IES_env import NR_IES_v0
import NR_IES
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
    # print(type(env.observation_space.sample()))
    #ValueError: ('Observation ({}) outside given space ({})!', array([12.5,  5. ,  2. ,  2.5,  2. ,  2. ]), Box(-10.0, 20.0, (6,), float32))
    print(env.observation_space.contains(np.array([12.5,  5. ,  2. ,  2.5,  2. ,  2. ])))
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