from collections import defaultdict
import pickle

import numpy as np

class Vocab(object):
  def __init__(self, dataset=None):
    self.word_to_index = {}
    self.index_to_word = {}
    self.word_freq = defaultdict(int)
    self.total_words = 0
    self.unknown = '<unk>'
    self.pad = '<pad>'
    self.eos = '<eos>'
    self.add_word(self.pad, count=0)
    self.add_word(self.unknown, count=0)
    self.add_word(self.eos, count=0)

    if dataset is not None:
    	self.construct(dataset)

  def add_word(self, word, count=1):
    if word not in self.word_to_index:
      index = len(self.word_to_index)
      self.word_to_index[word] = index
      self.index_to_word[index] = word
    self.word_freq[word] += count

  def construct(self, words):
    for word in words:
      self.add_word(word)
    self.total_words = float(sum(self.word_freq.values()))
    print '{} total words with {} uniques'.format(self.total_words, len(self.word_freq))

  def encode(self, word):
    if word not in self.word_to_index:
      word = self.unknown
    return self.word_to_index[word]

  def decode(self, index):
    return self.index_to_word[index]

  def save(self, path="./model/vocab/vocab.pkl"):
    pickle.dump(self, open(path, 'wb'))

  def load(path="./model/vocab/vocab.pkl"):
    return pickle.load(open(path, 'rb'))

  def __len__(self):
    return len(self.word_freq)