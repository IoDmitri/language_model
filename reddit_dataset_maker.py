from language_model import Language_model
from vocab import Vocab
import pandas as pd 

def gen_reddit_model():
	f_name = "./reddit_data/clean_data.txt"
	data = pd.read_csv(f_name)
	vocab = Vocab(flatten_np_array(data))
	model = Language_model(vocab=vocab, batch_size=150, max_steps=25, save_dir="reddit", num_layers=2)
	#model.train_on_file(f_name)
	model.train(data.values, save_path="./models/")

def flatten_np_array(arr):
	for r in arr:
		for word in r.split():
			yield word
		yield "<eos>"

if __name__ == "__main__":
	gen_reddit_model()
