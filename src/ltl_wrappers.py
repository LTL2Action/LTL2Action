"""
These are simple wrappers that will include LTL goals to any given environment.
It also progress the formulas as the agent interacts with the envirionment.

However, each environment must implement the followng functions:
    - *get_events(...)*: Returns the propositions that currently hold on the environment.
    - *sample_ltl_goal(...)*: Returns a new (randomly generated) LTL goal for the episode.

Notes about LTLEnv:
    - The episode ends if the LTL goal is progressed to True or False.
    - If the LTL goal becomes True, then an extra +1 reward is given to the agent.
    - If the LTL goal becomes False, then an extra -1 reward is given to the agent.
    - Otherwise, the agent gets the same reward given by the original environment.
"""


import numpy as np
import gym
from gym import spaces
import ltl_progression, random
from ltl_samplers import getLTLSampler, SequenceSampler

class LTLEnv(gym.Wrapper):
    def __init__(self, env, progression_mode="full", intrinsic=0.0):
        """
        LTL environment
        --------------------
        It adds an LTL objective to the current environment
            - The observations become a dictionary with an added "text" field
              specifying the LTL objective
            - It also automatically progress the formula and generates an
              appropriate reward function
            - However, it does requires the user to define a labeling function
              and a set of training formulas
        progression_mode:
            - "full": the agent gets the full, progressed LTL formula as part of the observation
            - "partial": the agent sees which propositions (individually) will progress or falsify the formula
            - "none": the agent gets the full, original LTL formula as part of the observation
        """
        super().__init__(env)
        self.progression_mode   = progression_mode
        self.observation_space = spaces.Dict({'features': env.observation_space})
        self.known_progressions = {}
        self.intrinsic = intrinsic

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
        self.known_progressions = {}
        self.obs = self.env.reset()

        # Defining an LTL goal
        self.ltl_goal     = self.sample_ltl_goal()
        self.ltl_original = self.ltl_goal

        # Adding the ltl goal to the observation
        if self.progression_mode == "partial":
            ltl_obs = {'features': self.obs,'progress_info': self.progress_info(self.ltl_goal)}
        else:
            ltl_obs = {'features': self.obs,'text': self.ltl_goal}
        return ltl_obs


    def step(self, action):
        int_reward = 0
        # executing the action in the environment
        next_obs, original_reward, env_done, info = self.env.step(action)

        # progressing the ltl formula
        truth_assignment = self.get_events(self.obs, action, next_obs)
        self.ltl_goal = self.progression(self.ltl_goal, truth_assignment)
        self.obs      = next_obs

        # Computing the LTL reward and done signal
        ltl_reward = 0.0
        ltl_done   = False
        if self.ltl_goal == 'True':
            ltl_reward = 1.0
            ltl_done   = True
        elif self.ltl_goal == 'False':
            ltl_reward = -1.0
            ltl_done   = True
        else:
            ltl_reward = int_reward

        # Computing the new observation and returning the outcome of this action
        if self.progression_mode == "full":
            ltl_obs = {'features': self.obs,'text': self.ltl_goal}
        elif self.progression_mode == "none":
            ltl_obs = {'features': self.obs,'text': self.ltl_original}
        elif self.progression_mode == "partial":
            ltl_obs = {'features': self.obs, 'progress_info': self.progress_info(self.ltl_goal)}
        else:
            raise NotImplementedError

        reward  = original_reward + ltl_reward
        done    = env_done or ltl_done
        return ltl_obs, reward, done, info

    def progression(self, ltl_formula, truth_assignment):

        if (ltl_formula, truth_assignment) not in self.known_progressions:
            result_ltl = ltl_progression.progress_and_clean(ltl_formula, truth_assignment)
            self.known_progressions[(ltl_formula, truth_assignment)] = result_ltl

        return self.known_progressions[(ltl_formula, truth_assignment)]


    # # X is a vector where index i is 1 if prop i progresses the formula, -1 if it falsifies it, 0 otherwise. 
    def progress_info(self, ltl_formula):
        propositions = self.env.get_propositions()
        X = np.zeros(len(self.propositions))

        for i in range(len(propositions)):
            progress_i = self.progression(ltl_formula, propositions[i])
            if progress_i == 'False':
                X[i] = -1.
            elif progress_i != ltl_formula:
                X[i] = 1. 
        return X

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
    def __init__(self, env, progression_mode="full", ltl_sampler=None, intrinsic=0.0):
        super().__init__(env, progression_mode, intrinsic)
        self.propositions = self.env.get_propositions()
        self.sampler = getLTLSampler(ltl_sampler, self.propositions)

    def sample_ltl_goal(self):
        # NOTE: The propositions must be represented by a char
        # This function must return an LTL formula for the task
        formula = self.sampler.sample()

        if isinstance(self.sampler, SequenceSampler):
            def flatten(bla):
                output = []
                for item in bla:
                    output += flatten(item) if isinstance(item, tuple) else [item]
                return output

            length = flatten(formula).count("and") + 1
            self.env.timeout = 25 # 10 * length

        return formula


    def get_events(self, obs, act, next_obs):
        # This function must return the events that currently hold on the environment
        # NOTE: The events are represented by a string containing the propositions with positive values only (e.g., "ac" means that only propositions 'a' and 'b' hold)
        return self.env.get_events()

