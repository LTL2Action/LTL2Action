import argparse
import time
import sys
import numpy as np
import glfw

import gym
import safety_gym
from gym import wrappers, logger
from envs.safety.safety_env import SafetyEnv

from safe_rl.utils.logx import EpochLogger

class RandomAgent(object):
    """This agent picks actions randomly"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        # return [1, 0]
        return self.action_space.sample()

class PlayAgent(object):
    """
    This agent allows user to play with Safety's Point agent.

    Use the UP and DOWN arrows to move forward and back and
    use '<' and '>' to rotate the agent.
    """
    def __init__(self, action_space):
        self.action_space = action_space
        self.prev_act = np.array([0, 0])

    def act(self, obs):
        key = obs.get('key_pressed', None)

        if(key == glfw.KEY_COMMA):
            current = np.array([0, 0.4])
        elif(key == glfw.KEY_PERIOD):
            current = np.array([0, -0.4])
        elif(key == glfw.KEY_UP):
            current = np.array([0.1, 0])
        elif(key == glfw.KEY_DOWN):
            current = np.array([-0.1, 0])
        elif(key == -1): # This is glfw.RELEASE
            current = np.array([0, 0])
            self.prev_act = np.array([0, 0])
        else:
            current = np.array([0, 0])

        self.prev_act = np.clip(self.prev_act + current, -1, 1)

        return self.prev_act

def run_policy(env_id, max_ep_len=None, num_episodes=100, render=True):
    # logger = EpochLogger()
    outdir = './storage/random-agent-results'

    env = gym.make(env_id)
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    if ("Play" in env_id):
        agent = PlayAgent(env.action_space)
    else:
        agent = RandomAgent(env.action_space)

    o, r, d, ep_ret, ep_cost, ep_len, n = env.reset(), 0, False, 0, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a = agent.act(o)
        a = np.clip(a, env.action_space.low, env.action_space.high)
        o, r, d, info = env.step(a)
        ep_ret += r
        ep_cost += info.get('cost', 0)
        ep_len += 1

        if d or (ep_len == max_ep_len):
            o, r, d, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0
            n += 1

    # logger.log_tabular('EpRet', with_min_and_max=True)
    # logger.log_tabular('EpCost', with_min_and_max=True)
    # logger.log_tabular('EpLen', average_only=True)
    # logger.dump_tabular()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='SafexpTest-v0', help='Select the environment to run')
    args = parser.parse_args()


    run_policy(args.env_id, max_ep_len=30000, num_episodes=1000)
