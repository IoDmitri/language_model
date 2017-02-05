import sys
import os

import numpy as np
import tensorflow as tf

class Language_model(object):
    def __init__(
        self, 
        vocab_size,
        restore=False, 
        num_layers=1, 
        device='gpu', 
        batch_size=64, 
        embed_size=100, 
        hidden_size=100, 
        dropout=0.90, 
        max_steps=45, 
        max_epochs=10, 
        lr=0.0001, 
        save_dir=None, 
        cell="gru"
    ):
        self._vocab_size=vocab_size
        self._num_layers = num_layers
        self._device = device
        self._batch_size = batch_size
        self._embed_size = embed_size
        self._hidden_size = hidden_size
        self._dropout = dropout
        self._max_steps = max_steps
        self._max_epochs = max_epochs
        self._lr = lr
        self._cell = cell
        self._add_placeholders()
        self._setup_graph()

    def _add_embedding(self):
        with tf.device(self._device + ":0"):
            embedding = tf.get_variable("Embedding", [self._vocab_size, self._embed_size], trainable=True)
            e_x = tf.nn.embedding_lookup(embedding, self.input_placeholder)
            return e_x

    def _add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32, shape=[None, self._max_steps])
        self.label_placeholder = tf.placeholder(tf.int32, shape=[None, self._max_steps])
        self.sequence_length = tf.placeholder(tf.int32, shape=[None])
        self._dropout_placeholder = tf.placeholder(tf.float32)

    def _gen_cell(self):
        if self._cell == "gru":
            return tf.nn.rnn_cell.GRUCell(self._hidden_size)
        elif self._cell == "lstm":
            return tf.nn.rnn_cell.LSTMCell(self._hidden_size, state_is_tuple=False)

    def _run_rnn(self, inputs):
        with tf.variable_scope("RNN") as scope:
            # embedded inputs are passed in here
            cell = self._gen_cell()
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self._dropout_placeholder)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self._num_layers, state_is_tuple=False)
            self.initial_state = cell.zero_state(self._batch_size, tf.float32)
            outputs, last_state = tf.nn.dynamic_rnn(
                cell = cell,
                inputs = inputs,
                sequence_length = self.sequence_length,
                initial_state = self.initial_state,
                scope=scope
            ) 

            return outputs, last_state

    def _projection_layer(self, rnn_ouputs):
        with tf.variable_scope("Projection") as scope:
            flattened = tf.reshape(rnn_ouputs, (-1, self._hidden_size), name="flattened")
            U = tf.get_variable("U", [self._hidden_size, self._vocab_size]) #len(self.vocab) + 1])
            b_2 = tf.get_variable("B", [self._vocab_size])
            outputs = tf.matmul(flattened, U) + b_2       
            return outputs

    def _compute_loss(self,projected_outputs):
        y_flat = tf.reshape(self.label_placeholder, [-1])
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(projected_outputs, y_flat)
        mask = tf.sign(tf.to_float(y_flat))
        masked_loss = mask * losses

        masked_loss = tf.reshape(masked_loss, tf.shape(self.label_placeholder))

        mean_loss_by_example = tf.reduce_sum(masked_loss, reduction_indices=1) / (tf.to_float(self.sequence_length) + 1e-12 )
        mean_loss = tf.reduce_mean(mean_loss_by_example)
        tf.add_to_collection("total_loss", mean_loss)
        loss = tf.add_n(tf.get_collection("total_loss"))
        return loss

    def _add_train_step(self, loss):
        opt = tf.train.AdamOptimizer(self._lr)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 1)
        return opt.apply_gradients(zip(grads, tvars))

    def _setup_graph(self):
        print "initializing model"
        self.inputs = self._add_embedding()
        self.rnn_ouputs, self.final_state = self._run_rnn(self.inputs)
        self.outputs = self._projection_layer(self.rnn_ouputs)
        self.predictions = tf.nn.softmax(tf.cast(self.outputs, "float64"), name="predictions")
        self.loss_op = self._compute_loss(self.outputs)
        self.trainOp = self._add_train_step(self.loss_op)
