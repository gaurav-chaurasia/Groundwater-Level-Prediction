from flask import Flask, request, jsonify
from torch_utils import get_prediction

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # load the data from user
    pm = request.args['p']
    tmx = request.args['tmax']
    tmn = request.args['tmin']
    
    prediction = get_prediction(p=pm, tmax=tmx, tmin=tmn)



    # convert text to tensor
    # prediction
    # return the json data
    return jsonify({'result': True})

