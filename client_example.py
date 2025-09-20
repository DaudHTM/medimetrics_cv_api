import requests

URL = "http://127.0.0.1:8000/measure-hand"
IMAGE_PATH = "test1.png"

with open(IMAGE_PATH, "rb") as f:
    files = {"file": (IMAGE_PATH, f, "image/png")}
    r = requests.post(URL, files=files)
    print(r.status_code)
    print(r.json())
