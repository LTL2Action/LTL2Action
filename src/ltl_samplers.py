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


# This class generates random LTL formulas using the following template:
#   ('until',('not','a'),('and', 'b', ('until',('not','c'),'d')))
# where p1, p2, p3, and p4 are randomly sampled propositions
class DefaultSampler(LTLSampler):
    def sample(self):
        p = random.sample(self.propositions,4)
        return ('until',('not',p[0]),('and', p[1], ('until',('not',p[2]),p[3])))

# This class generates random LTL formulas that form a sequence of actions.
# @ min_len, max_len: min/max length of the random sequence to generate.
class SequenceSampler(LTLSampler):
    def __init__(self, propositions, min_len=2, max_len=4):
        super().__init__(propositions)
        self.min_len = int(min_len)
        self.max_len = int(max_len)

    def sample(self):
        seq = "".join(random.sample(self.propositions, random.randint(self.min_len, self.max_len)))
        ret = self._get_sequence(seq)

        return ret

    def _get_sequence(self, seq):
        if len(seq) == 1:
            return ('until','True',seq)
        return ('until','True', ('and', seq[0], self._get_sequence(seq[1:])))

# The LTLSampler factory method that instantiates the proper sampler
# based on the @sampler_id.
def getLTLSampler(sampler_id, propositions):
    tokens = ["Default"]
    if (sampler_id != None):
        tokens = sampler_id.split("_")

    if (tokens[0] == "Sequence"):
        return SequenceSampler(propositions, tokens[1], tokens[2])
    else: # "Default"
        return DefaultSampler(propositions)

