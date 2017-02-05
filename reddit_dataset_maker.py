import sys
import re

from language_model import Language_model
from vocab import Vocab
from model_trainer import Model_Trainer
from config import Config
from text_generator import Text_Generator 

num_layers = 3
hidden_size = 250
embed_size= 250
max_epochs = 60
dropout= 0.90

def gen_reddit_model():
	f_name = "./reddit_data/clean_data.txt"
	#f_name = "merged_data.txt"
	v_file_name = "./reddit_data/valid.txt"
	conf = Config(
		num_layers=num_layers,
		batch_size=175,
		max_steps=25,
		hidden_size=hidden_size,
		embed_size=embed_size,
		max_epochs=max_epochs,
		dropout=dropout
	)
	trainer = Model_Trainer(f_name, v_file_name, config=conf, save_dir="./models/reddit", min_count=10)
	trainer.fit()
	# model = Language_model(batch_size=175, max_steps=25, save_dir="reddit", num_layers=num_layers, hidden_size=hidden_size, embed_size=embed_size, max_epochs=max_epochs, dropout=dropout, min_count=10)
	# model.train_on_file(f_name, v_file_name)
	#model.export_latest_model()
def reddit_gen_text():
	conf = Config(batch_size=1, num_layers=num_layers, max_steps=1, hidden_size=hidden_size, embed_size=embed_size)
	gen = Text_Generator("./models/reddit", conf)
	starting_text = "once upon a time"
        while starting_text:
            print ' '.join(gen.generate_sentence(starting_text, 25))
            starting_text = raw_input(">")
	# model = Language_model(batch_size=1, max_steps=1, save_dir="reddit", num_layers=num_layers, hidden_size=hidden_size, embed_size=embed_size)
	# model.gen_text_shell()

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
