from flask import Flask, request, jsonify,
import tensorflow as tf

from config import Config
from text_generator import Text_Generator

app = Flask(__name__, template_folder="./")

conf = Config(batch_size=1, num_layers=3, max_steps=1, hidden_size=250, embed_size=250)
gen = Text_Generator("./models/reddit", conf)

@app.route("/gen_text", methods=["POST"])
def gen_text():
	initial_text = request.get_json(force=True)["text"]
	final_text = " ".join(gen.generate_sentence(initial_text, 25)[:-1])
	jsonify({"text" : final_text})
