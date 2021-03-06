from flask import Flask


app = Flask(__name__)


@app.route('/')
def home():
    return 'Hello! This is an api to get generated music with a model trained using Deep Learning techniques. It was built as part of a project in Deep Learning course offered in 2021 at Innopolis University.'


@app.route('/generate')
def generate():
    # generate music with your model & return
    return "Here's a generated music...Enjoy!"
