if __name__ == '__main__':
    # This is a terrible hack just to be able to execute this file directly
    import sys
    sys.path.insert(0, '../../')

import random, math, os
import numpy as np
import gym
from gym import spaces
from envs.minigrid.adversarial import *

class MinigridEnv(gym.Env):
    """
    A simple wrapper for a gym-minigrid environment. This implements propositions on top of the minigrid. 
    """

    def __init__(self, env, letters, timeout = 100):
        """
            ## env is the wrapped MiniGrid environment
        """
        self.env = env
        self.letters = letters
        self.letter_types = list(set(letters))
        self.letter_types.sort()
        self.action_space = env.action_space
        self.observation_space = env.observation_space['image']
        self.num_episodes = 0
        self.time = 0
        self.timeout = timeout

    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        self.time += 1
        if self.time >= self.timeout:
            done = True
        return obs['image'], 0, done, _

    def seed(self, seed=None):
        random.seed(seed)
        self.env.seed(seed)

    def reset(self):
        """
        This function resets the world and collects the first observation.
        """
        self.num_episodes += 1
        self.time  = 0
        return self.env.reset()['image']

    def get_events(self):
        return self.env.get_events()

    def get_propositions(self):
        return self.letter_types

class AdversarialMinigridEnv(MinigridEnv):
    def __init__(self):
        super().__init__(AdversarialEnv9x9(), 'abc', 1000)

if __name__ == '__main__':
    AdversarialMinigridEnv()

