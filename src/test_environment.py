"""
This code allows to play the environment manually.
To control the agent, use the WASD keys.
NOTE:
    Letter-5x5-v0 -> Standard environment of 5x5 with a timeout of 150 steps
    Letter-5x5-v1 -> This version uses a fixed map of 5x5 with a timeout of 150 steps
    Letter-5x5-v2 -> Standard environment of 5x5 using an agent-centric view with a timeout of 150 steps
    Letter-5x5-v3 -> This version uses a fixed map of 5x5 using an agent-centric view with a timeout of 150 steps
"""


import gym
import envs.gym_letters
import ltl_wrappers


def test_env():
    env = gym.make("Letter-7x7-v0")
    #env = ltl_wrappers.LTLLetterEnv(env, ltl_sampler="Sequence_2_3")
    env = ltl_wrappers.LTLLetterEnv(env, ltl_sampler="UntilTasks_1_3_1_2")
    str_to_action = {"w":0,"s":1,"a":2,"d":3}

    import random
    for _ in range(10):
        obs = env.reset()
        for _ in range(10000):
            env.show()
            print(obs["ltl"])
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



if __name__ == '__main__':

    test_env()
