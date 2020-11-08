"""
This code allows to play the environment manually.
We support two environments: 1. Simple-LTL-Env-v0 (Default): The LTL formula is the env and actions are the propositions. The goal is to progress the formula ot True.
                             2. Letter_envs of different sizes along with the goal specified in LTL. To control the agent, use the WASD keys.

NOTE:
    Letter-5x5-v0 -> Standard environment of 5x5 with a timeout of 150 steps
    Letter-5x5-v1 -> This version uses a fixed map of 5x5 with a timeout of 150 steps
    Letter-5x5-v2 -> Standard environment of 5x5 using an agent-centric view with a timeout of 150 steps
    Letter-5x5-v3 -> This version uses a fixed map of 5x5 using an agent-centric view with a timeout of 150 steps
"""


import argparse

import gym
import envs.gym_letters
import ltl_wrappers


def test_env(env, sampler):
    env = gym.make(env)
    #env = ltl_wrappers.LTLLetterEnv(env, progression_mode="full", ltl_sampler="Sequence_2_3")
    env = ltl_wrappers.LTLLetterEnv(env, progression_mode="full", ltl_sampler=sampler)
    str_to_action = {"w":0,"s":1,"a":2,"d":3}

    import random
    for _ in range(10):
        obs = env.reset()
        for _ in range(10000):
            env.show()
            print(obs["text"])
            print("\nAction? ", end="")
            a = input()
            while a not in str_to_action:
                a = input()
            print()
            a = str_to_action[a]
            #a = random.randrange(env.action_space.n)
            obs, reward, done, info = env.step(a%env.action_space.n)

            if done:
                env.show()
                print(reward)
                print("Done!")
                input()
                break

            print(reward)

    env.close()

def test_simple_ltl_env(env, sampler):
    env = gym.make(env)
    env = ltl_wrappers.LTLLetterEnv(env, progression_mode="full", ltl_sampler=sampler)

    letter_types = env.propositions

    def valid_action(a):
        return a in letter_types

    for _ in range(10):
        obs = env.reset()
        for _ in range(10000):
            env.show()
            print(obs["text"])
            print("\nAction? ", end="")
            a = input().strip()
            while not valid_action(a):
                a = input().strip()
            a = letter_types.index(a)

            obs, reward, done, info = env.step(a)

            if done:
                env.show()
                print(reward)
                print("Done!")
                input()
                break

            print(reward)

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Simple-LTL-Env-v0",
                    help="name of the environment to train on (default: Simple-LTL-Env-v0)")
    parser.add_argument("--ltl-sampler", default="UntilTasks_2_2_1_1",
                    help="the ltl formula template to sample from (default: UntilTasks_2_2_1_1)")
    args = parser.parse_args()


    test_simple_ltl_env(args.env, args.ltl_sampler)
