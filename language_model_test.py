import numpy as np
import tensorflow as tf
from language_model import Language_model

def test_model():
	model = Language_model(vocab=range(0,100))
	s = tf.Session()
	#1 more than step size to acoomodate for the <eos> token at the end
	random_data = np.random.normal(0, 100, size=[42068, 46])
	# file = "./data/ptb.test.txt"
	print "Fitting started"
	model.train(random_data, s)

if __name__ == "__main__":
	test_model()