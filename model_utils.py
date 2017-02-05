
import numpy as np
import tensorflow as tf

from config import Config
from language_model import Language_model

def restore_model(path, session, saver):
	saver.restore(session, tf.train.latest_checkpoint(path))

def create_model(vocab_size, config):
	return Language_model(
			vocab_size=vocab_size,
			num_layers=config.num_layers,
			device=config.device,
			batch_size=config.batch_size, 
			embed_size=config.embed_size, 
			hidden_size=config.hidden_size, 
			dropout=config.dropout, 
			max_steps=config.max_steps, 
			max_epochs=config.max_epochs, 
			lr=config.lr,  
			cell=config.cell)

