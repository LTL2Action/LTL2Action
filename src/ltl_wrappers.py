import numpy as np
import gym
from gym import spaces
import ltl_progression
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

        self.env.show()

        # Defining an LTL goal
        self.ltl_goal = self.sample_ltl_goal()

        # Adding the ltl goal to the observation
        ltl_obs = {'features': self.obs,'ltl': self.ltl_goal}

        return ltl_obs


    def step(self, action):
        # executing the action in the environment
        next_obs, original_reward, done, info = self.env.step(action)

        # progressing the ltl formula
        truth_assignment = self.get_events(self.obs, action, next_obs)
        self.ltl_goal = ltl_progression.progress(self.ltl_goal, truth_assignment)
        self.obs      = next_obs

        # Computing the LTL reward and done signal
        ltl_reward = 0
        ltl_done   = False
        if self.ltl_goal == 'True':
            ltl_reward = 1
            ltl_done   = True
        if self.ltl_goal == 'False':
            ltl_reward = -1
            ltl_done   = True

        # Computing the new observation and returning the outcome of this action
        ltl_obs = {'features': self.obs,'ltl': self.ltl_goal}
        return ltl_obs, original_reward + ltl_reward, done or ltl_done, info



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

    def sample_ltl_goal(self):
        # This function must return an LTL formula for the task
        # Format:     
        #(
        #    'and',
        #    ('until','True', ('and', 'd', ('until','True',('not','c')))),
        #    ('until','True', ('and', 'a', ('until','True', ('and', 'b', ('until','True','c')))))
        #)
        # NOTE: The propositions must be represented by a char
        return ('until',('not','a'),'c')

    def get_events(self, obs, act, next_obs):
        # This function must return the events that currently hold on the environment
        # NOTE: The events are represented by a string containing the propositions with positive values only (e.g., "ac" means that only propositions 'a' and 'b' hold)
        return self.env.get_events()
