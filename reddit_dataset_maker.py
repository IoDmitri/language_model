from language_model import Language_model 

def gen_reddit_model():
	model = Language_model(batch_size=150, max_steps=25, save_dir="reddit", num_layers=2)
	f_name = "./reddit_data/data.txt"
	model.train_on_file(f_name)

if __name__ == "__main__":
	gen_reddit_model()
