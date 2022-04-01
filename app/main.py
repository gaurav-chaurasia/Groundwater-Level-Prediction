from flask import Flask, request, jsonify, render_template
from app.torch_utils import get_prediction

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # load the data from user
    pm = request.form['p']
    tmx = request.form['tmax']
    tmn = request.form['tmin']
    
    prediction = get_prediction(p=pm, tmax=tmx, tmin=tmn)



    # convert text to tensor
    # prediction
    # return the json data
    # return jsonify({'result': True, 'prediction': prediction})
    return render_template('prediction.html', prediction=prediction)
