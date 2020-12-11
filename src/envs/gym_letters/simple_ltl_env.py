import random, math, os
import numpy as np
import gym
from gym import spaces

class SimpleLTLEnv(gym.Env):

    def __init__(self, letters:str, timeout:int):
        """
            letters:
                - (str) propositions
            timeout:
                - (int) maximum lenght of the episode
        """
        self.letters      = letters
        self.letter_types = list(set(letters))
        self.letter_types.sort()
        self.action_space = spaces.Discrete(len(self.letter_types))
        self.observation_space = spaces.Discrete(1)
        self.num_episodes = 0
        self.time = 0
        self.timeout = timeout
        self.proposition = None

    def step(self, action):
        """
        This function executes an action in the environment.
        """
        self.time += 1
        reward = 0.0
        done = self.time > self.timeout
        obs = self._get_observation()
        self.proposition = action

        return obs, reward, done, {}

    def _get_observation(self):
        return self.observation_space.sample()

    def seed(self, seed=None):
        return

    def reset(self):
        self.time = 0
        self.num_episodes += 1
        obs = self._get_observation()

        return obs

    def show(self):
        print("Events:", self.get_events(), "\tTimeout:", self.timeout - self.time)

    def get_events(self):
        return self.letter_types[self.proposition] if self.proposition != None else None

    def get_propositions(self):
        return self.letter_types

class SimpleLTLEnvDefault(SimpleLTLEnv):
    def __init__(self):
        super().__init__(letters="abcdefghijkl", timeout=75)
