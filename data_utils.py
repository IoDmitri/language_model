import numpy as np
import tensorflow as tf

def data_iterator(data, batch_size, max_length_size):
	#data is a generator that will generate data, it assumnes that each generation will be a list that contains the sentence and ends with <eos>
	batch_data = np.zeros([batch_size, max_length_size])
	labels = np.zeros([batch_size, max_length_size])
	sizes = np.zeros([batch_size])
	i = 1
	for ex in data:
		#print "ex - {0}".format(ex)
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

def multigen(gen_func):
    class _multigen(object):
        def __init__(self, *args, **kwargs):
            self.__args = args
            self.__kwargs = kwargs
        def __iter__(self):
            return gen_func(*self.__args, **self.__kwargs)
    return _multigen

@multigen
def process_file_data(f_name, process_fn=None, max_sent_len=None, flatten=False):
	with open(f_name) as file:
		for line in file:
			if flatten:
				for word in line.split():
					yield word
				yield "<eos>"
			else:
				 for sent in _partition_sentence_by_batch_size([process_fn(x) if process_fn else x for x in (line.split() + ['<eos>'])], max_sent_len):
				 	yield sent

def _partition_sentence_by_batch_size(sentence, max_sent_len):
	if not max_sent_len or len(sentence) <= max_sent_len:
		yield sentence
	else:
		total_bins = len(sentence) / max_sent_len
		for i in range(0, len(sentence), max_sent_len):
			yield sentence[i: i+max_sent_len]

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    # from https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

def gen_dataset(dataset, fn):
	return [fn(x) for x in process_file_data(dataset, flatten=True)]