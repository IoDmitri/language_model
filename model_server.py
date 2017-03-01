from __future__ import print_function # In python 2.7
import sys
import os

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import tensorflow as tf

from config import Config
from data_utils import normalize_text
from text_generator import Text_Generator

app = Flask(__name__)
app.debug=False
CORS(app)

conf = Config(batch_size=1, num_layers=3, max_steps=1, hidden_size=250, embed_size=250, device="cpu")
gen = Text_Generator("./models/wiki", conf)

@app.route("/")
def hello():
	return "Hello World"

@app.route("/gen_text", methods=["POST"])
def gen_text():
	print("request - {0}".format(request), file=sys.stderr)
	print ("json - {0}".format(request.get_json(force=True)), file=sys.stderr)
	json = request.get_json(force=True)
	initial_text = request.get_json(force=True)["text"]
	cleaned_text = normalize_text(initial_text, lower=False)
	final_text = " ".join(gen.generate_sentence(cleaned_text, 25)[:-1])
	return jsonify([{"text" : final_text}])

if __name__ == '__main__':
	port = os.environ.get("PORT", "5000")
	print("App running on port - {0}".format(port), file=sys.stderr)
	app.run(host="0.0.0.0", port=port)
