from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # load the data from user
    q = request.args['q']
    tmin = request.args['tmin']
    tmax = request.args['tmax']
    print(q, tmin, tmax)
    
    # convert text to tensor
    # prediction
    # return the json data
    return jsonify({'result': True})

