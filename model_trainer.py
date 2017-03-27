import sys
import os

import numpy as np
import tensorflow as tf

from config import Config
from data_utils import *
from model_utils import *
from language_model import Language_model
from vocab import Vocab

class Model_Trainer(object):
	def __init__(self, fname, validation = None, config=None, save_dir=None, restore_path=None, min_count=None, verbose=10):
		self.vocab =  None
		if save_dir and os.path.exists(save_dir + "/" + "vocab.pkl"):
			print "found vocab at {0}".format(save_dir + "/" + "vocab.pkl")
			self.vocab = Vocab.load(path = save_dir + "/" + "vocab.pkl")
		else: 
			self.vocab = Vocab(process_file_data(fname, flatten=True), min_count = min_count)
			self.vocab.save(save_dir + "/" + "vocab.pkl")
			print "saved vocab to {0}".format(save_dir + "/" + "vocab.pkl")

		self.config = config if config else Config()
		self._model = create_model(len(self.vocab) + 1, config)

		self.save_dir=save_dir
		self.restore_path=restore_path
		self.verbose=verbose
		self._data = process_file_data(fname, process_fn=self.vocab.encode, max_sent_len=self.config.max_steps, in_token_form=config.in_token_form)
		self._validation_set = process_file_data(validation, process_fn=self.vocab.encode, max_sent_len=self.config.max_steps, in_token_form=config.in_token_form) if validation else None
	
	def fit(self, save=True):
		saver = tf.train.Saver()

		#for summary
		pp_tensor = tf.exp(tf.reduce_mean(self._model.loss_op))
		tf.summary.scalar("preplexity",  pp_tensor)

		tf.summary.histogram("rnn_outputs", self._model.rnn_ouputs)
		tf.summary.histogram("final_state", self._model.final_state)

		summary_op = tf.summary.merge_all()

		writer = tf.summary.FileWriter(self.save_dir, graph=tf.get_default_graph())

		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
			if self.restore_path:
				restore_model(self.restore_path, sess, saver)
			else :
				start = tf.global_variables_initializer()
				sess.run(start)

			for epoch in xrange(self.config.max_epochs):
				train_pp = self._run_epoch(self._model, self._data, sess, self._model.trainOp, self.verbose, saver, summary_op, writer, epoch)
				print "Training preplexity for batch {} - {}".format(epoch, train_pp)
				if self._validation_set:
					validation_pp = self._run_epoch(self._model, self._validation_set, sess, verbose=self.verbose)
				print "Validation preplexity for batch  {} - {}".format(epoch, validation_pp)

				if save:
					self._save_model(sess, saver, False)

	def _run_epoch(self, model, data, sess, trainOp=None, verbose=10, saver=None, summaryOp=None, writer=None, epoch=0):
		drop = self.config.dropout
		if not trainOp:
			trainOp = tf.no_op()
			drop = 1

		if summaryOp is None:
			summaryOp = tf.no_op()

		total_steps = sum(1 for x in data_iterator(data, self.config.batch_size, self.config.max_steps))
		state = model.initial_state.eval()
		train_loss = []
		for step, (x,y, l) in enumerate(data_iterator(data, self.config.batch_size, self.config.max_steps)):
			feed = {
				model.input_placeholder: x,
				model.label_placeholder: y,
				model.sequence_length: l,
				model._dropout_placeholder: drop,
				model.initial_state: state
			}
			loss, state, _, summary = sess.run([model.loss_op, model.final_state, trainOp, summaryOp], feed_dict=feed)
			train_loss.append(loss)
			
			if summary and writer:
				writer.add_summary(summary, step*(epoch+1)

			if verbose and step % verbose == 0:
				sys.stdout.write('\r{} / {} : pp = {}'. format(step, total_steps, np.exp(np.mean(train_loss))))
				sys.stdout.flush()
				if saver:
					self._save_model(sess, saver)
			if verbose:
				sys.stdout.write('\r')

		return np.exp(np.mean(train_loss))

	def _save_model(self, sess, saver, quiet=True):
		save_path = self.save_dir + "/" + self.config.name
		if not quiet:
			print "saving model to {0}".format(save_path)
		saver.save(sess, save_path)
		if not quiet:
			print "saved model"

