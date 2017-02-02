from flask import Flask, request, jsonify
import tensorflow as tf

from language_model import Language_model
from vocab import Vocab

app = Flask(__name__)

sess = tf.Session()

model = Language_model(
	save_dir="reddit",
	num_layers=3,
	hidden_size=250,
	embed_size=250
)

model.restore(session=sess)

@app.route("/gen_text", methods=["POST"]):
	initial_text = request.form["text"]
	final_text = model.generate_sentence(starting_text, 25, sess)
	return jsonify(result=[{"text" : final_text}])

	