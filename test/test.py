import requests

input_data={
    'P': 1,
    'Tmax': 2,
    'Tmin': 3
}
# 2014-12-30	0.062500	4.756	-10.230	8771.143378

resp = requests.post(url="http://localhost:5000/predict?p=0.062500&tmin=-10.230&tmax=4.756")

print(resp.text)