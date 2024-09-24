import requests

url = 'http://127.0.0.1:5000/predict'
file_path = r"C:\Users\riya2\Downloads\AUDIO\REAL\trump-original.wav"

with open(file_path, 'rb') as file:
    files = {'file': file}
    response = requests.post(url, files=files)

print(response.json())
