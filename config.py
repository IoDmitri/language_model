class Config(object):
	def __init__(
		self, 
		num_layers=1, 
		device='gpu', 
		batch_size=64, 
		embed_size=100, 
		hidden_size=100, 
		dropout=0.90, 
		max_steps=45, 
		max_epochs=10, 
		lr=0.0001,
		cell="gru",
		name="language_model",
		in_token_form=False
	   ):
		self.num_layers=num_layers
		self.device=device
		self.batch_size=batch_size
		self.embed_size=embed_size
		self.hidden_size=hidden_size
		self.dropout=dropout
		self.max_epochs=max_epochs
		self.max_steps=max_steps
		self.lr=lr
		self.cell=cell
		self.name="language_model"
		self.in_token_form=in_token_form