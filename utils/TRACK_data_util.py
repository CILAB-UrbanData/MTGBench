import os
import math
import time
import json
import torch
import pickle
import random
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from torch.utils.data import Dataset, DataLoader

class WordVocab:
    def __init__(self, traj_path, seq_len=128):
        self.pad_index = 0
        self.unk_index = 1
        self.sos_index = 2  # CLS
        self.mask_index = 3
        self.eos_index = 4
        specials = ["<pad>", "<unk>", "<sos>", "<mask>", "<eos>"]

        train = pd.read_csv(traj_path, sep=';')

        counter = Counter()
        paths = train['path'].values
        for i in tqdm(range(paths.shape[0]), desc='location counting'):
            path_l = eval(paths[i])[:seq_len]
            for p in path_l:
                counter[p] += 1
        self.freqs = counter

        self.index2loc = list(specials)
        for tok in specials:
            del counter[tok]

        ## sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        self.del_edge = []
        self.all_edge = []
        for word, freq in words_and_frequencies:
            if freq < 1:
                self.del_edge.append(word)
                continue
            self.index2loc.append(word)
            self.all_edge.append(word)
        self.loc2index = {tok: i for i, tok in enumerate(self.index2loc)}
        self.vocab_size = len(self.index2loc)

        users = train['usr_id'].values
        users = np.unique(users)
        specials = ["<pad>", "<unk>"]
        self.index2usr = list(specials)
        for u in users:
            self.index2usr.append(u)
        self.usr2index = {tok: i for i, tok in enumerate(self.index2usr)}
        self.user_num = len(self.index2usr)

    def to_seq(self, sentence, seq_len=None, with_eos=False, with_sos=False, with_len=False):
        if isinstance(sentence, str):
            sentence = sentence.split()

        seq = [self.loc2index.get(word, self.unk_index) for word in sentence]

        if with_sos:
            seq = [self.sos_index] + seq

        origin_seq_len = len(seq)

        if seq_len is not None:
            if len(seq) <= seq_len:
                seq += [self.pad_index for _ in range(seq_len) - len(seq)]
            else:
                seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_len else seq

    def from_seq(self, seq, join=False, with_pad=False):
        words = [self.index2loc[idx]
                 if idx < len(self.index2loc)
                 else "<%d>" % idx
                 for idx in seq
                 if not with_pad or idx != self.pad_index]

        return " ".join(words) if join else words

    @staticmethod
    def load_vocab(vocab_path: str) -> 'WordVocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)

    def __eq__(self, other):
        if self.loc2index != other.loc2index:
            return False
        if self.index2loc != other.index2loc:
            return False
        return True

    def __len__(self):
        return len(self.index2loc)

    def vocab_rerank(self):
        self.loc2index = {word: i for i, word in enumerate(self.index2loc)}

    def extend(self, v, sort=False):
        words = sorted(v.index2loc) if sort else v.index2loc
        for w in words:
            if w not in self.loc2index:
                self.index2loc.append(w)
                self.loc2index[w] = len(self.index2loc) - 1

class AbstractDataset(object):

    def __init__(self, config):
        raise NotImplementedError("Dataset not implemented")

    def get_data(self):
        raise NotImplementedError("get_data not implemented")

    def get_data_feature(self):
        raise NotImplementedError("get_data_feature not implemented")
