import sys
import re

from language_model import Language_model
from vocab import Vocab

def gen_reddit_model():
	f_name = "./reddit_data/clean_data.txt"
	v_file_name = "./reddit_data/valid.txt"
	model = Language_model(batch_size=100, max_steps=25, save_dir="reddit", num_layers=3, max_epochs=10, min_count=10)
	model.train_on_file(f_name, v_file_name)

def reddit_gen_text():
	model = Language_model(batch_size=1, max_steps=1, save_dir="reddit", num_layers=3)
	model.gen_text_shell()

def flatten_np_array(arr):
	for r in arr:
		for sentence in r:
			for word in sentence:
				yield word
			yield "<eos>"

if __name__ == "__main__":

	z = gen_reddit_model if len(sys.argv) == 1 else reddit_gen_text
	z() 
