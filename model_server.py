from flask import Flask, request, jsonify
import tensorflow as tf

from language_model import Language_model
from vocab import Vocab

app = Flask(__name__)

sess = tf.Session()

model = Language_model(
	batch_size=1,
	max_steps=1,
	save_dir="reddit",
	num_layers=3,
	hidden_size=250,
	embed_size=250
)

model.restore(session=sess)

@app.route("/gen_text", methods=["POST"])
def gen_text():
	initial_text = request.get_json(force=True)["text"]
	final_text = " ".join(model.generate_sentence(initial_text, 25, sess)[:-1])
	return jsonify(result=[{"text" : final_text}])

	