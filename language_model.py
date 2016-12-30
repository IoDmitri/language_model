import sys

import numpy as np
import tensorflow as tf

from tensorflow.python.ops.seq2seq import sequence_loss
from data_utils import data_iterator, process_file_data
# from tensorflow.nn.rnn_cell import GRUCell, DropoutWrapper


class Language_model(object):

    def __init__(self, vocab=None, device='gpu', batch_size=64, embed_size=100, hidden_size=100, dropout=0.90, max_steps=45, max_epochs=20, lr=0.001):
        self._device = device
        self._batch_size = batch_size
        self._embed_size = embed_size
        self._hidden_size = hidden_size
        self._dropout = dropout
        self._max_steps = max_steps
        self._max_epochs = max_epochs
        self._lr = lr
        self.vocab = vocab
        self._is_initialized = False
        self._add_placeholders()

    def _add_embedding(self):
        with tf.device(self._device + ":0"):
            embedding = tf.get_variable("Embedding", [len(self.vocab), self._embed_size], trainable=True)
            e_x = tf.nn.embedding_lookup(embedding, self.input_placeholder)
            return e_x

    def _add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32, shape=[None, self._max_steps])
        self.label_placeholder = tf.placeholder(tf.int32, shape=[None, self._max_steps])
        self.sequence_length = tf.placeholder(tf.int32, shape=[None])
        self._dropout_placeholder = tf.placeholder(tf.float32)

    def _run_rnn(self, inputs):
        # embedded inputs are passed in here
        self.initial_state = tf.zeros([self._batch_size, self._hidden_size], tf.float32)
        cell = tf.nn.rnn_cell.GRUCell(self._hidden_size)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self._dropout_placeholder)

        outputs, last_state = tf.nn.dynamic_rnn(
            cell = cell,
            inputs = inputs,
            sequence_length = self.sequence_length,
            initial_state = self.initial_state
        ) 

        return outputs, last_state

    def _projection_layer(self, rnn_ouputs):
        #outputs will be off size [batch x max_step x hidden_size]
        rnn_ouputs = [tf.squeeze(s, [1]) for s in tf.split(1, self._max_steps, rnn_ouputs)] 
        with tf.variable_scope("Projection") as scope:
            U = tf.get_variable("U", [self._hidden_size, len(self.vocab)])
            b_2 = tf.get_variable("B", [len(self.vocab)])
            outputs = [tf.matmul(x, U) + b_2 for x in rnn_ouputs]

        return outputs

    def _compute_loss(self,projected_outputs):
        projected_outputs = tf.reshape(tf.concat(1, projected_outputs), [-1, len(self.vocab)])
        ones = [tf.ones([self._batch_size * self._max_steps], tf.float32)]
        seq_loss = sequence_loss(
            [projected_outputs], 
            [tf.reshape(self.label_placeholder, [-1])], 
            ones
        )
        print "Sequence loss - {0}".format(seq_loss)
        tf.add_to_collection('total_loss', seq_loss)
        loss = tf.add_n(tf.get_collection('total_loss')) 
        print "Loss - {0}".format(loss)
        return loss

    def _add_train_step(self, loss):
        opt = tf.train.AdamOptimizer(self._lr)
        return opt.minimize(loss)

    def _run_epoch(self, data, session, inputs, rnn_ouputs, loss, trainOp, verbose=10):
        with session.as_default() as sess:
            total_steps = sum(1 for x in data_iterator(data, self._batch_size, self._max_steps))
            train_loss = []
            for step, (x,y, l) in enumerate(data_iterator(data, self._batch_size, self._max_steps)):
                print "step - {0}".format(step)
                feed = {
                    self.input_placeholder: x,
                    self.label_placeholder: y,
                    self.sequence_length: l,
                    self._dropout_placeholder: self._dropout,
                }
                _, loss = sess.run([trainOp, loss], feed_dict=feed)
                print "loss - {0}".format(loss)
                train_loss.append(loss)
                if verbose and step % verbose == 0:
                    sys.stdout.write('\r{} / {} : pp = {}'. format(step, total_steps, np.exp(np.mean(train_loss))))
                    sys.stdout.flush()
                if verbose:
                    sys.stdout.write('\r')

            return np.exp(np.mean(train_loss))


    def train(self,data, session=tf.Session(), verbose=10):

        print "initializing model"
        self._add_placeholders()
        inputs = self._add_embedding()
        rnn_ouputs, _ = self._run_rnn(inputs)
        outputs = self._projection_layer(rnn_ouputs)
        loss = self._compute_loss(outputs)
        trainOp = self._add_train_step(loss)
        start = tf.initialize_all_variables()
        saver = tf.train.Saver()

        with session as sess:
            sess.run(start)

            for epoch in xrange(self._max_epochs):
                train_pp = self._run_epoch(data, sess, inputs, rnn_ouputs, loss, trainOp, verbose)
                print "Training preplexity for batch {} - {}".format(epoch, train_pp)


    def _encode_dataset(self, data):
        return [self.vocab.encode(x) for x in data]

    def _build_vocab(self,data):
        pass
        # TO DO- implement with Vocab class
            

                

