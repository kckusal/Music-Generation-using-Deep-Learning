from flask import Flask
from flask_cors import CORS, cross_origin


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/')
def home():
    return 'Hello! This is an api to get generated music with a model trained using Deep Learning techniques. It was built as part of a project in Deep Learning course offered in 2021 at Innopolis University.'


@app.route('/generate')
@cross_origin()
def generate():
    # generate music with your model & return
    return "Here's a generated music...Enjoy!"
