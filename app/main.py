from flask import Flask, request, jsonify, render_template
from torch_utils import get_groundwater_prediction

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/model')
def model():
    return render_template('model_exp.html')

@app.route('/predict', methods=['POST'])
def predict():
    # load the data from user
    Month = request.form['Month']
    Irrigation = request.form['Irrigation']
    Rainfall = request.form['Rainfall']
    Tem = request.form['Tem']
    Evaporation = request.form['Evaporation']
    Collection_Date = request.form['date']
    print(Collection_Date)
    prediction = get_groundwater_prediction(m=Month, i=Irrigation, r=Rainfall, t=Tem, e=Evaporation)



    # convert text to tensor
    # prediction
    # return the json data
    # return jsonify({'result': True, 'prediction': prediction})
    return render_template('prediction.html', prediction=prediction)
