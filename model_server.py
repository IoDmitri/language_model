from flask import Flask, request, jsonify,
import tensorflow as tf

from config import Config
from data_utils import normalize_text
from text_generator import Text_Generator

app = Flask(__name__, template_folder="./")

conf = Config(batch_size=1, num_layers=3, max_steps=1, hidden_size=250, embed_size=250)
gen = Text_Generator("./models/reddit", conf)

@app.route("/gen_text", methods=["POST"])
def gen_text():
	initial_text = request.get_json(force=True)["text"]
	cleaned_text = normalize_text(initial_text)
	final_text = " ".join(gen.generate_sentence(cleaned_text, 25)[:-1])
	jsonify({"text" : final_text})
