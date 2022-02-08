import requests

input_data={
    'Q': 1,
    'Tmax': 2,
    'Tmin': 3
}

resp = requests.post(url="http://localhost:5000/predict?q=1&tmin=2&tmax=3")

print(resp.text)