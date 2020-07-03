"""
This code allows to play the environment manually.
To control the agent, use the WASD keys.
"""


import gym
import envs.gym_letters
import ltl_wrappers


def test_env():
    env = gym.make("Letter-4x4-v0")
    env = ltl_wrappers.LTLLetterEnv(env)
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
