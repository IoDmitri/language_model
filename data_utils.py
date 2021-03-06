import re
import string

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
		ex_size = len(ex)
		idx = i-1
		batch_data[idx, 0:ex_size -1] = ex[0:-1]
		labels[idx, 0:ex_size-1] = ex[1:]
		sizes[idx] = ex_size
		i +=1

	#at this point we can't fill any more of the batch, so yeild what we've got
	yield batch_data, labels, sizes

def ptb_iterator(raw_data, batch_size, num_steps):
  # Pulled from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py#L82
  raw_data = np.array(raw_data, dtype=np.int32)
  data_len = len(raw_data)
  batch_len = data_len // batch_size
  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
  epoch_size = (batch_len - 1) // num_steps
  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
  for i in range(epoch_size):
    x = data[:, i * num_steps:(i + 1) * num_steps]
    y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]

    yield (x, y, [num_steps * batch_size])

def normalize_text(text, replace_br=True, replace_punct=True, lower=True):
	norm_text = text
	if lower:
		norm_text = text.lower()
	norm_text = norm_text.strip()
	norm_text = re.sub('\d', "N", norm_text)

	# Replace breaks with spaces
	if replace_br:
		norm_text = norm_text.replace('<br />', ' ')

	if replace_punct:
		# Pad punctuation with spaces on both sides
		for char in string.punctuation + "`":
		    norm_text = norm_text.replace(char, ' ' + char + ' ')

	return norm_text

def multigen(gen_func):
    class _multigen(object):
        def __init__(self, *args, **kwargs):
            self.__args = args
            self.__kwargs = kwargs
        def __iter__(self):
            return gen_func(*self.__args, **self.__kwargs)
    return _multigen

@multigen
def process_file_data(f_name, process_fn=None, max_sent_len=None, flatten=False, in_token_form=False):
	with open(f_name) as file:
		for line in file:
			if in_token_form:
				line = normalize_text(line, replace_br=False, replace_punct=False)
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

def gen_dataset(dataset, fn):
	return [fn(x) for x in process_file_data(dataset, flatten=True)]