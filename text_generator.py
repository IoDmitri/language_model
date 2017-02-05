import numpy as np 
import tensorflow as tf

from language_model import Language_model
from model_utils import *
from vocab import Vocab

class Text_Generator(object):
	def __init__(self, restore_path, config):
		self.vocab = Vocab.load(path = restore_path + "/" + "vocab.pkl")
		self._model = create_model(len(self.vocab) + 1, config)
		self._sess = tf.Session()
		
		saver = tf.train.Saver()
		restore_model(restore_path, self._sess, saver)

	def _generate_text(self, starting_text='<eos>',stop_length=100, stop_tokens=None, session=None, temp=1.0):
		#self._maybe_initialize(sess)
		with session.as_default():
			state = self._model.initial_state.eval()
			# Imagine tokens as a batch size of one, length of len(tokens[0])
			tokens = [self.vocab.encode(word) for word in starting_text.split()]

			#prime the network over our inputed sentence
			for token in tokens[:-1]:
				state, y_pred = session.run(
					[self._model.final_state, self._model.predictions[-1]], feed_dict= {
						self._model.input_placeholder : [tokens[-1:]],
						self._model.initial_state: state,
						self._model._dropout_placeholder: 1,
						self._model.sequence_length: [1] 
					}
				)

			for i in xrange(stop_length):
				state, y_pred = session.run(
					[self._model.final_state, self._model.predictions[-1]], feed_dict= {
						self._model.input_placeholder : [tokens[-1:]],
						self._model.initial_state: state,
						self._model._dropout_placeholder: 1,
						self._model.sequence_length: [1] 
					}
				)
				next_word_idx = self._sample(y_pred, temperature=temp)
				tokens.append(next_word_idx)
				if stop_tokens and self.vocab.decode(tokens[-1]) in stop_tokens:
					break
			output = [self.vocab.decode(word_idx) for word_idx in tokens]
			return output

	def generate_sentence(self, starting_text, stop_length=100):
		"""Convenice to generate a sentence from the model."""
		return self._generate_text(starting_text, stop_tokens=['<eos>', '<pad>'], stop_length=stop_length, session=self._sess)

	def _sample(self, a, temperature=1.0):
	    # helper function to sample an index from a probability array
	    # from https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
	    a = np.log(a) / temperature
	    a = np.exp(a) / np.sum(np.exp(a))
	    return np.argmax(np.random.multinomial(1, a, 1))

