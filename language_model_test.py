import numpy as np
import tensorflow as tf
from language_model import Language_model
from vocab import Vocab

def test_model():
	model = Language_model(vocab=range(0,101))
	s = tf.Session()
	#1 more than step size to acoomodate for the <eos> token at the end
	random_data = np.random.randint(0, 101, size=[42068, 46])
	# file = "./data/ptb.test.txt"
	print "Fitting started"
	model.train(random_data, s)

def test_on_ptb_test_datast():
	test_data_file = "./data/ptb.train.txt"
	validation_data_file = "./data/ptb.valid.txt"
	model = Language_model(max_steps=10, embed_size=150, num_layers=2)
	model.train_on_file(test_data_file, validation_data_file)

def train_and_validate_save():
	test_data_file = "./data/ptb.train.txt"
	validation_data_file = "./data/ptb.valid.txt"
	model = Language_model(max_steps=10, embed_size=150, num_layers=2)
	model.train_on_file(test_data_file, validation_data_file)

	



if __name__ == "__main__":
	test_on_ptb_test_datast()