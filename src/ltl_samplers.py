"""
This class is responsible for sampling LTL formulas typically from
given template(s).

@ propositions: The set of propositions to be used in the sampled
                formula at random.
"""

import random

class LTLSampler():
    def __init__(self, propositions):
        self.propositions = propositions

    def sample(self):
        raise NotImplementedError


# Samples from one of the other samplers at random. The other samplers are sampled by their default args.
class SuperSampler(LTLSampler):
    def __init__(self, propositions):
        super().__init__(propositions)
        self.reg_samplers = getRegisteredSamplers(self.propositions)

    def sample(self):
        return random.choice(self.reg_samplers).sample()

# This class samples formulas of form (or, op_1, op_2), where op_1 and 2 can be either specified as samplers_ids
# or by default they will be sampled at random via SuperSampler.
class OrSampler(LTLSampler):
    def __init__(self, propositions, sampler_ids = ["SuperSampler"]*2):
        super().__init__(propositions)
        self.sampler_ids = sampler_ids

    def sample(self):
        return ('or', getLTLSampler(self.sampler_ids[0], self.propositions).sample(),
                        getLTLSampler(self.sampler_ids[1], self.propositions).sample())

# This class generates random LTL formulas using the following template:
#   ('until',('not','a'),('and', 'b', ('until',('not','c'),'d')))
# where p1, p2, p3, and p4 are randomly sampled propositions
class DefaultSampler(LTLSampler):
    def sample(self):
        p = random.sample(self.propositions,4)
        return ('until',('not',p[0]),('and', p[1], ('until',('not',p[2]),p[3])))

# This class generates random conjunctions of Until-Tasks.
# Each until tasks has *n* levels, where each level consists
# of avoiding a proposition until reaching another proposition.
#   E.g.,
#      Level 1: ('until',('not','a'),'b')
#      Level 2: ('until',('not','a'),('and', 'b', ('until',('not','c'),'d')))
#      etc...
# The number of until-tasks, their levels, and their propositions are randomly sampled.
# This code is a generalization of the DefaultSampler---which is equivalent to UntilTaskSampler(propositions, 2, 2, 1, 1)
class UntilTaskSampler(LTLSampler):
    def __init__(self, propositions, min_levels=1, max_levels=2, min_conjunctions=1 , max_conjunctions=2):
        super().__init__(propositions)
        self.levels       = (int(min_levels), int(max_levels))
        self.conjunctions = (int(min_conjunctions), int(max_conjunctions))
        assert 2*int(max_levels)*int(max_conjunctions) <= len(propositions), "The domain does not have enough propositions!"

    def sample(self):
        # Sampling a conjuntion of *n_conjs* (not p[0]) Until (p[1]) formulas of *n_levels* levels
        n_conjs = random.randint(*self.conjunctions)
        p = random.sample(self.propositions,2*self.levels[1]*n_conjs)
        ltl = None
        b = 0
        for i in range(n_conjs):
            n_levels = random.randint(*self.levels)
            # Sampling an until task of *n_levels* levels
            until_task = ('until',('not',p[b]),p[b+1])
            b +=2
            for j in range(1,n_levels):
                until_task = ('until',('not',p[b]),('and', p[b+1], until_task))
                b +=2
            # Adding the until task to the conjunction of formulas that the agent have to solve
            if ltl is None: ltl = until_task
            else:           ltl = ('and',until_task,ltl)
        return ltl


# This class generates random LTL formulas that form a sequence of actions.
# @ min_len, max_len: min/max length of the random sequence to generate.
class SequenceSampler(LTLSampler):
    def __init__(self, propositions, min_len=2, max_len=4):
        super().__init__(propositions)
        self.min_len = int(min_len)
        self.max_len = int(max_len)

    def sample(self):
        length = random.randint(self.min_len, self.max_len)
        seq = ""

        while len(seq) < length:
            c = random.choice(self.propositions)
            if len(seq) == 0 or seq[-1] != c:
                seq += c

        ret = self._get_sequence(seq)

        return ret

    def _get_sequence(self, seq):
        if len(seq) == 1:
            return ('eventually',seq)
        return ('eventually',('and', seq[0], self._get_sequence(seq[1:])))

class AdversarialEnvSampler(LTLSampler):
    def sample(self):
        p = random.randint(0,1)
        if p == 0:
            return ('eventually', ('and', 'a', ('eventually', 'b')))
        else:
            return ('eventually', ('and', 'a', ('eventually', 'c')))

def getRegisteredSamplers(propositions):
    return [SequenceSampler(propositions),
            UntilTaskSampler(propositions),
            DefaultSampler(propositions)]

# The LTLSampler factory method that instantiates the proper sampler
# based on the @sampler_id.
def getLTLSampler(sampler_id, propositions):
    tokens = ["Default"]
    if (sampler_id != None):
        tokens = sampler_id.split("_")

    # Don't change the order of ifs here otherwise the OR sampler will fail
    if (tokens[0] == "OrSampler"):
        return OrSampler(propositions)
    elif ("_OR_" in sampler_id): # e.g., Sequence_2_4_OR_UntilTask_3_3_1_1
        sampler_ids = sampler_id.split("_OR_")
        return OrSampler(propositions, sampler_ids)
    elif (tokens[0] == "Sequence"):
        return SequenceSampler(propositions, tokens[1], tokens[2])
    elif (tokens[0] == "UntilTasks"):
        return UntilTaskSampler(propositions, tokens[1], tokens[2], tokens[3], tokens[4])
    elif (tokens[0] == "SuperSampler"):
        return SuperSampler(propositions)
    elif (tokens[0] == "AdversarialSampler"):
        return AdversarialEnvSampler(propositions)
    else: # "Default"
        return DefaultSampler(propositions)

