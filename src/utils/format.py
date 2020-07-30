"""
These functions preprocess the observations.
When trying more sophisticated encoding for LTL, we might have to modify this code.
"""

import os
import json
import re
import torch
import torch_ac
import gym
import numpy as np
import dgl
import networkx as nx
from sklearn.preprocessing import OneHotEncoder

import utils


def get_obss_preprocessor(obs_space, vocab_space, gnn):
    # Check if obs_space is an image space
    if isinstance(obs_space, gym.spaces.Box):
        obs_space = {"image": obs_space.shape}

        def preprocess_obss(obss, device=None):
            return torch_ac.DictList({
                "image": preprocess_images(obss, device=device)
            })

    # Check if it is a MiniGrid observation space
    elif isinstance(obs_space, gym.spaces.Dict) and list(obs_space.spaces.keys()) == ["features"]:
        obs_space = {"image": obs_space.spaces["features"].shape, "text": len(vocab_space) + 9}
        vocab_space = {"max_size": obs_space["text"], "tokens": vocab_space}

        vocab = Vocabulary(vocab_space)
        def preprocess_obss(obss, device=None):
            return torch_ac.DictList({
                "image": preprocess_images([obs["features"] for obs in obss], device=device),
                "text":  preprocess_texts([obs["text"] for obs in obss], vocab, vocab_space, gnn=gnn, device=device)
            })
        preprocess_obss.vocab = vocab

    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss


def preprocess_images(images, device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = np.array(images)
    return torch.tensor(images, device=device, dtype=torch.float)


def preprocess_texts(texts, vocab, vocab_space, gnn=False, device=None):
    if (gnn):
        return preprocess4gnn(texts, vocab, vocab_space, device)

    return preprocess4rnn(texts, vocab, device)


def preprocess4rnn(texts, vocab, device=None):
    """
    This function receives the LTL formulas and convert them into inputs for an RNN
    """
    var_indexed_texts = []
    max_text_len = 0

    for text in texts:
        text = str(text) # transforming the ltl formula into a string
        tokens = re.findall("([a-z]+)", text.lower())
        var_indexed_text = np.array([vocab[token] for token in tokens])
        var_indexed_texts.append(var_indexed_text)
        max_text_len = max(len(var_indexed_text), max_text_len)

    indexed_texts = np.zeros((len(texts), max_text_len))

    for i, indexed_text in enumerate(var_indexed_texts):
        indexed_texts[i, :len(indexed_text)] = indexed_text

    return torch.tensor(indexed_texts, device=device, dtype=torch.long)

def preprocess4gnn(texts, vocab, vocab_space, device=None):
    """
    This function receives the LTL formulas and convert them into inputs for a GNN
    """
    propositions = vocab_space["tokens"]
    ret = np.array([[to_graph(text, vocab, propositions)] for text in texts])
    # print(ret)
    return ret

def to_graph(formula, vocab, propositions):
    PARENT, SIBLING  = "par_child", "sibling"
    # A helper function that recursively builds up the AST of the LTL formula
    def to_graph_(nxg, formula, node_id):
        head = formula[0]
        rest = formula[1:]

        if head in ["next", "until", "and", "or"]:
            nxg.add_node(node_id, feat=vocab.one_hot(head), type=head)

            id_l = to_graph_(nxg, rest[0], node_id + 1)
            nxg.add_edge(node_id + 1, node_id, type=PARENT)
            id_r = to_graph_(nxg, rest[1], id_l + 1)
            nxg.add_edge(id_l + 1, node_id, type=PARENT)

            return id_r

        if head in ["eventually", "always", "not"]:
            nxg.add_node(node_id, feat=vocab.one_hot(head), type=head)

            id_c = to_graph_(nxg, rest[0], node_id + 1)
            nxg.add_edge(node_id + 1, node_id, type=PARENT)

            return id_c

        if formula in ["True", "False"]:
            nxg.add_node(node_id, feat=vocab.one_hot(formula), type=formula)

            return node_id

        if formula in propositions:
            nxg.add_node(node_id, feat=vocab.one_hot(formula.replace("'",'')), type=formula)

            return node_id


        assert False, "Format error in to_graph_()"

        return None

    nxg = nx.DiGraph()
    to_graph_(nxg, formula, 0)

    # concert the Networkx graph to dgl graph and pass the 'feat' attribute
    g = dgl.DGLGraph()
    g.from_networkx(nxg, node_attrs=["feat"]) #edge_attrs=["type"]

    return g


class Vocabulary:
    """A mapping from tokens to ids with a capacity of `max_size` words.
    It can be saved in a `vocab.json` file."""

    def __init__(self, vocab_space):
        self.max_size = vocab_space["max_size"]
        self.vocab = {}
        # # populate the vocab with the LTL operators
        # for item in ['next', 'until', 'and', 'or', 'eventually', 'always', 'not']:
        #     self.__getitem__(item)

        self.one_hot = self.make_encoders(vocab_space["tokens"])

    """ This function prepares the encoder that for te propositions.
        For now we only support 1-hot but others (positional?) encoders can be added to this.
    """
    def make_encoders(self, propositions):
        terminals = ['True', 'False'] + propositions
        enc = OneHotEncoder(handle_unknown='ignore', dtype=np.int)
        enc.fit([['next'], ['until'], ['and'], ['or'], ['eventually'],
            ['always'], ['not']] + np.array(terminals).reshape((-1, 1)).tolist())

        def _one_hot(token):
            return enc.transform([[token]])[0][0].toarray()

        return _one_hot

    def load_vocab(self, vocab):
        self.vocab = vocab

    def __getitem__(self, token):
        if not token in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]
