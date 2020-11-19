if __name__ == '__main__':
    # This is a terrible hack just to be able to execute this file directly
    import sys
    sys.path.insert(0, '../../')

import random, math, os
import numpy as np
import gym
from gym import spaces
from gym_minigrid.envs.adversarial import *

class MinigridEnv(gym.Env):
    """
    A simple wrapper for a gym-minigrid environment. This implements propositions on top of the minigrid. 
    """

    def __init__(self, env, letters):
        """
            ## env is the wrapped MiniGrid environment
        """
        self.letters = letters
        self.letter_types = list(set(letters))
        self.letter_types.sort()
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.num_episodes = 0

    def step(self, action):
        return self.env.step(action)

    def seed(self, seed=None):
        random.seed(seed)
        env.random.seed(seed)

    def reset(self):
        """
        This function resets the world and collects the first observation.
        """
        self.num_episodes += 1

        return env.reset()

    def get_events(self):
        return env.get_events()

    def get_propositions(self):
        return self.letter_types

class AdversarialMinigridEnv(MinigridEnv):
    def __init__(self):
        super().__init__(AdversarialEnv10x10())

if __name__ == '__main__':
    AdversarialMinigridEnv()

