import sys
import os

import numpy as np
import tensorflow as tf

from tensorflow.python.ops.seq2seq import sequence_loss
from data_utils import *
from vocab import Vocab
# from tensorflow.nn.rnn_cell import GRUCell, DropoutWrapper


class Language_model(object):
    def __init__(self, vocab=None, session=None, num_layers = 1, device='gpu', batch_size=64, embed_size=100, hidden_size=100, dropout=0.90, max_steps=45, max_epochs=10, lr=0.001, save_dir=None, min_count=None):
        self._num_layers = num_layers
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
        self._current_session= session if session is not None else tf.Session()
        self._name = "language_model"
        self._save_dir = save_dir
        self._min_count = min_count

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
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self._num_layers)

        outputs, last_state = tf.nn.dynamic_rnn(
            cell = cell,
            inputs = inputs,
            sequence_length = self.sequence_length,
            initial_state = self.initial_state
        ) 

        return outputs, last_state

    def _projection_layer(self, rnn_ouputs):
        with tf.variable_scope("Projection") as scope:
            flattened = tf.reshape(rnn_ouputs, (-1, self._hidden_size), name="flattened")
            U = tf.get_variable("U", [self._hidden_size, len(self.vocab)])
            b_2 = tf.get_variable("B", [len(self.vocab)])
            outputs = tf.matmul(flattened, U) + b_2       
            return outputs

    def _compute_loss(self,projected_outputs):
        ones = [tf.ones([self._batch_size * self._max_steps], tf.float32)]
        seq_loss = sequence_loss(
            [projected_outputs], 
            [tf.reshape(self.label_placeholder, [-1])], 
            ones
        )
        tf.add_to_collection('total_loss', seq_loss)
        loss = tf.add_n(tf.get_collection('total_loss')) 
        return loss

    def _add_train_step(self, loss):
        opt = tf.train.AdamOptimizer(self._lr)
        return opt.minimize(loss)

    def _run_epoch(self, data, session, trainOp=None, verbose=10):
        with session.as_default() as sess:
            drop = self._dropout
            if not trainOp:
                trainOp = tf.no_op()
                drop = 1
            total_steps = sum(1 for x in data_iterator(data, self._batch_size, self._max_steps))
            state = self.initial_state.eval()
            train_loss = []
            for step, (x,y, l) in enumerate(data_iterator(data, self._batch_size, self._max_steps)):
                feed = {
                    self.input_placeholder: x,
                    self.label_placeholder: y,
                    self.sequence_length: l,
                    self._dropout_placeholder: drop,
                    self.initial_state: state
                }
                loss, state, _ = sess.run([self.loss_op, self.final_state, trainOp], feed_dict=feed)
                train_loss.append(loss)
                if verbose and step % verbose == 0:
                    sys.stdout.write('\r{} / {} : pp = {}'. format(step, total_steps, np.exp(np.mean(train_loss))))
                    sys.stdout.flush()
                if verbose:
                    sys.stdout.write('\r')

            return np.exp(np.mean(train_loss))

    def _setup_graph(self):
        print "initializing model"
        self.inputs = self._add_embedding()
        self.rnn_ouputs, self.final_state = self._run_rnn(self.inputs)
        self.outputs = self._projection_layer(self.rnn_ouputs)
        self.predictions = tf.nn.softmax(tf.cast(self.outputs, "float64"))
        self.loss_op = self._compute_loss(self.outputs)
        self.trainOp = self._add_train_step(self.loss_op)

    def train(self,data,verbose=10, validation_set=None, save_path=None):
        if not self.vocab:
            self.vocab = Vocab(data, min_count = self._min_count)

        self._setup_graph()

        start = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with self._current_session as sess:
            self._maybe_initialize(sess)

            for epoch in xrange(self._max_epochs):
                train_pp = self._run_epoch(data, sess, self.trainOp, verbose)
                print "Training preplexity for batch {} - {}".format(epoch, train_pp)
                if validation_set:
                    validation_pp = self._run_epoch(validation_set, sess, verbose=verbose)
                    print "Validation preplexity for batch  {} - {}".format(epoch, validation_pp)

            if save_path:
                if self._save_dir:
                    save_path += self._save_dir + "/"
                print "saving model to {0}".format(save_path)
                saver.save(sess, save_path + self._name)
                print "saved model"
                vocab_path = save_path + "vocab.pkl"
                print "saving vocab to {0}".format(vocab_path)
                vocab.save(path=vocab_path)
                print "vocab saved"

    def restore(self, path=None, model_name=None, session=None):
        if not session:
            session = self._current_session

        model_name = model_name if model_name else self._name
        path = path if path else "./models/" 
        if self._save_dir:
            path += self._save_dir + "/"
        self.vocab = Vocab.load(path=path + "vocab.pkl")
        self._setup_graph()
        full_path = path + model_name + ".meta"
        print "full path - {0}".format(full_path)
        restorer = tf.train.import_meta_graph(full_path)
        restorer.restore(session, tf.train.latest_checkpoint(path))


    def train_on_file(self, fname, validation_fname=None, save_path="./models/"):
        if self._save_dir:
            save_path += self._save_dir + "/"
        self.vocab = Vocab(process_file_data(fname, flatten=True), min_count = self._min_count)
        validation_gen = None   
        if validation_fname:
            validation_gen = process_file_data(validation_fname, process_fn=self.vocab.encode, max_sent_len=self._max_steps)

        self.train(process_file_data(fname, process_fn=self.vocab.encode, max_sent_len=self._max_steps), validation_set=validation_gen, save_path=save_path)


    def generate_text(self, starting_text='<eos>',stop_length=100, stop_tokens=None, session=None, temp=1.0):
        if session is None:
            session = self._current_session
        else:
            session = session.as_default()

        with session as sess:
            self._maybe_initialize(sess)
            state = self.initial_state.eval()
            # Imagine tokens as a batch size of one, length of len(tokens[0])
            tokens = [self.vocab.encode(word) for word in starting_text.split()]

            #prime the network over our inputed sentence
            for token in tokens[:-1]:
                state, y_pred = self._current_session.run(
                    [self.final_state, self.predictions[-1]], feed_dict= {
                        self.input_placeholder : [tokens[-1:]],
                        self.initial_state: state,
                        self._dropout_placeholder: self._dropout,
                        self.sequence_length: [1] 
                    }
                )

            for i in xrange(stop_length):
                state, y_pred = self._current_session.run(
                    [self.final_state, self.predictions[-1]], feed_dict= {
                        self.input_placeholder : [tokens[-1:]],
                        self.initial_state: state,
                        self._dropout_placeholder: self._dropout,
                        self.sequence_length: [1] 
                    }
                )
                next_word_idx = sample(y_pred, temperature=temp)
                tokens.append(next_word_idx)
                if stop_tokens and self.vocab.decode(tokens[-1]) in stop_tokens:
                    break
            output = [self.vocab.decode(word_idx) for word_idx in tokens]
            return output

    def generate_sentence(self, starting_text, stop_length, session=None):
        """Convenice to generate a sentence from the model."""
        return self.generate_text(starting_text, stop_length, stop_tokens=['<eos>', '<pad>'], session=session)

    def _maybe_initialize(self, session):
        if not self._is_initialized:
            start = tf.global_variables_initializer()
            with session.as_default() as sess:
                sess.run(start)
                self._is_initialized = True

    def gen_text_shell(self):
        with tf.variable_scope("gen_text") as scope:
            with self._current_session as sess:
                self.restore(session=sess)
                starting_text = "once upon a time"
                while starting_text:
                    print ' '.join(self.generate_sentence(starting_text, 15, sess))
                    starting_text = raw_input(">")            



