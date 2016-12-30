import numpy as np
import tensorflow as tf

def data_iterator(data, batch_size, max_length_size):
	#data is a generator that will generate data, it assumnes that each generation will be a list that contains the sentence and ends with <eos>
	batch_data = np.zeros([batch_size, max_length_size])
	labels = np.zeros([batch_size, max_length_size])
	sizes = np.zeros([batch_size])
	i = 1
	for ex in data:
		#single example
		if i > batch_size:
			yield batch_data, labels, sizes
			i = 1
			batch_data = np.zeros([batch_size, max_length_size])
			labels = np.zeros([batch_size, max_length_size])
			sizes = np.zeros([batch_size])
		#numpy has 0 based arrays, and the last value from data should be an <eos> token
		ex_size = len(ex)-1
		idx = i-1
		batch_data[idx, 0:ex_size] = ex[0:-1]
		labels[idx, 0:ex_size] = ex[1:]
		sizes[idx] = ex_size - 1
		i +=1

	#at this point we can't fill any more of the batch, so yeild what we've got
	yield batch_data, labels, sizes

def process_file_data(f_name, process_fn=None):
	with open(f_name) as file:
		for line in file:
			yield [process_fn(x) if process_fn else x for x in (line.split() + ['<eos>'])] 