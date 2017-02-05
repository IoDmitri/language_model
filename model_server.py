from flask import Flask, request, jsonify
import tensorflow as tf

from config import Config
from text_generator import Text_Generator
from vocab import Vocab

app = Flask(__name__)

conf = Config(batch_size=1, num_layers=num_layers, max_steps=1, hidden_size=hidden_size, embed_size=embed_size)
gen = Text_Generator("./models/reddit", conf)

@app.route("/gen_text", methods=["POST"])
def gen_text():
	initial_text = request.get_json(force=True)["text"]
	final_text = " ".join(gen.generate_sentence(initial_text, 25)[:-1])
	return jsonify(result=[{"text" : final_text}])

	