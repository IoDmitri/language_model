import sys
import re

from language_model import Language_model
from vocab import Vocab

num_layers = 1
hidden_size = 100
embed_size= 100
max_epochs = 6
def gen_reddit_model():
	#f_name = "./reddit_data/clean_data.txt"
	f_name = "merged_data.txt"
	v_file_name = "./reddit_data/valid.txt"
	model = Language_model(batch_size=100, max_steps=25, save_dir="reddit", num_layers=num_layers, hidden_size=hidden_size, embed_size=embed_size, max_epochs=max_epochs, min_count=10)
	model.train_on_file(f_name, v_file_name)

def reddit_gen_text():
	model = Language_model(batch_size=1, max_steps=1, save_dir="reddit", num_layers=num_layers, hidden_size=hidden_size, embed_size=embed_size)
	model.gen_text_shell()

def test():
	model = Language_model(batch_size=1, max_steps=1, save_dir="reddit", num_layers=num_layers, hidden_size=hidden_size, embed_size=embed_size)
	#model2 = Language_model(batch_size=1, max_steps=1, save_dir="reddit", num_layers=num_layers, hidden_size=hidden_size, embed_size=embed_size)
	#assert(model.test() == model2.test())
	print model.test()

def flatten_np_array(arr):
	for r in arr:
		for sentence in r:
			for word in sentence:
				yield word
			yield "<eos>"

if __name__ == "__main__":
	#test()
	z = gen_reddit_model if len(sys.argv) == 1 else reddit_gen_text
	z() 
