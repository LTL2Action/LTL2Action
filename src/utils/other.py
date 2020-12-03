import random
import numpy
import torch
import collections


def seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synthesize(array):
    d = collections.OrderedDict()
    d["mean"] = numpy.mean(array)
    d["std"] = numpy.std(array)
    d["min"] = numpy.amin(array)
    d["max"] = numpy.amax(array)
    return d


def average_reward_per_step(returns, num_frames):
    avgs = []
    assert(len(returns) == len(num_frames))

    for i in range(len(returns)):
        avgs.append(returns[i] / num_frames[i])

    return numpy.mean(avgs)


def average_discounted_return(returns, num_frames, disc):
    discounted_returns = []
    assert(len(returns) == len(num_frames))

    for i in range(len(returns)):
        discounted_returns.append(returns[i] * (disc ** (num_frames[i]-1)))

    return numpy.mean(discounted_returns)