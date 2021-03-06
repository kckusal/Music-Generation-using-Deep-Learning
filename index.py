from flask import Flask


app = Flask(__name__)


@app.route('/')
def home():
    return 'This is an api to get generated music with a model trained using Deep Learning techniques.'


@app.route('/generate')
def generate():
    # generate music with your model & return
    return "Here's a generated music...Enjoy!"
