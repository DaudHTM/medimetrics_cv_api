import requests
import base64

URL = "http://127.0.0.1:8000/measure-face"
IMAGE_PATH = "facetst.png"

def main():
    with open(IMAGE_PATH, "rb") as f:
        files = {"file": (IMAGE_PATH, f, "image/png")}
        r = requests.post(URL, files=files)

    print("Status:", r.status_code)
    print("Content-Type:", r.headers.get("content-type"))
    print("Response preview (first 2000 chars):")
    print(r.text[:2000])

    try:
        data = r.json()
    except Exception as ex:
        print("Response is not JSON:", ex)
        # save raw response for inspection
        with open("last_response.txt", "wb") as out:
            out.write(r.content)
        print("Saved raw response to last_response.txt")
        return

    print("Parsed JSON:")
    print(data)

    # measurements (may be {} if no detections)
    measurements = data.get("measurements")
    print("measurements:", measurements)

    # save JSON response for inspection
    try:
        with open("last_face_response.json.txt", "w", encoding="utf-8") as jf:
            import json
            json.dump(data, jf, indent=2)
        print("Saved JSON response to last_face_response.json.txt")
    except Exception as e:
        print("Failed to save JSON response:", e)

    img_b64 = data.get("annotated_image_b64")
    if img_b64:
        img_bytes = base64.b64decode(img_b64)
        out_path = "annotated_face.png"
        with open(out_path, "wb") as out:
            out.write(img_bytes)
        print(f"Annotated image saved to {out_path}")

if __name__ == "__main__":
    main()
