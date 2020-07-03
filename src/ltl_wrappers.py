import numpy as np
import gym
from gym import spaces
import ltl_progression, random
#from collections import deque
#from baselines.common.atari_wrappers import LazyFrames


class LTLEnv(gym.Wrapper):
    def __init__(self, env):
        """
        LTL environment
        --------------------
        It adds an LTL objective to the current environment
            - The observations become a dictionary with an added "ltl" field
              specifying the LTL objective
            - It also automatically progress the formula and generates an 
              appropriate reward function
            - However, it does requires the user to define a labeling function
              and a set of training formulas
        """
        super().__init__(env)
        self.observation_space = spaces.Dict({'features': env.observation_space})
        self.known_progressions = {}

    def sample_ltl_goal(self):
        # This function must return an LTL formula for the task
        # Format:     
        #(
        #    'and',
        #    ('until','True', ('and', 'd', ('until','True',('not','c')))),
        #    ('until','True', ('and', 'a', ('until','True', ('and', 'b', ('until','True','c')))))
        #)
        # NOTE: The propositions must be represented by a char
        raise NotImplementedError

    def get_events(self, obs, act, next_obs):
        # This function must return the events that currently hold on the environment
        # NOTE: The events are represented by a string containing the propositions with positive values only (e.g., "ac" means that only propositions 'a' and 'b' hold)
        raise NotImplementedError

    def reset(self):
        self.obs = self.env.reset()

        # Defining an LTL goal
        self.ltl_goal = self.sample_ltl_goal()

        # Adding the ltl goal to the observation
        ltl_obs = {'features': self.obs,'ltl': self.ltl_goal}

        return ltl_obs


    def step(self, action):
        # executing the action in the environment
        next_obs, original_reward, env_done, info = self.env.step(action)

        # progressing the ltl formula
        truth_assignment = self.get_events(self.obs, action, next_obs)
        if (self.ltl_goal, truth_assignment) not in self.known_progressions:
            self.known_progressions[(self.ltl_goal, truth_assignment)] = ltl_progression.progress_and_clean(self.ltl_goal, truth_assignment)
        self.ltl_goal = self.known_progressions[(self.ltl_goal, truth_assignment)]
        self.obs      = next_obs

        # Computing the LTL reward and done signal
        ltl_reward = 0.0
        ltl_done   = False
        if self.ltl_goal == 'True':
            ltl_reward = 1.0
            ltl_done   = True
        if self.ltl_goal == 'False':
            ltl_reward = -1.0
            ltl_done   = True

        # Computing the new observation and returning the outcome of this action
        ltl_obs = {'features': self.obs,'ltl': self.ltl_goal}
        reward  = original_reward + ltl_reward
        done    = env_done or ltl_done
        return ltl_obs, reward, done, info


class IgnoreLTLWrapper(gym.Wrapper):
    def __init__(self, env):
        """
        Removes the LTL formula from an LTLEnv
        It is useful to check the performance of off-the-shelf agents
        """
        super().__init__(env)
        self.observation_space =  env.observation_space['features']

    def reset(self):
        obs = self.env.reset()
        obs = obs['features']
        return obs

    def step(self, action):
        # executing the action in the environment
        obs, reward, done, info = self.env.step(action)
        obs = obs['features']
        return obs, reward, done, info 


class LTLLetterEnv(LTLEnv):
    def __init__(self, env):
        """
        LTL environment
        --------------------
        It adds an LTL objective to the current environment
            - The observations become a dictionary with an added "ltl" field
              specifying the LTL objective
            - It also automatically progress the formula and generates an 
              appropriate reward function
            - However, it does requires the user to define a labeling function
              and a set of training formulas
        """
        super().__init__(env)
        self.propositions = self.env.get_propositions()

    def sample_ltl_goal(self):
        # NOTE: The propositions must be represented by a char
        # This function must return an LTL formula for the task
        # We generate random LTL formulas using the following template:
        #    ('until',('not','a'),('and', 'b', ('until',('not','c'),'d')))
        # where p1, p2, p3, and p4 are randomly sampled propositions
        p = random.sample(self.propositions,4)
        return ('until',('not',p[0]),('and', p[1], ('until',('not',p[2]),p[3])))

    def get_events(self, obs, act, next_obs):
        # This function must return the events that currently hold on the environment
        # NOTE: The events are represented by a string containing the propositions with positive values only (e.g., "ac" means that only propositions 'a' and 'b' hold)
        return self.env.get_events()

