import requests
import base64

URL = "http://127.0.0.1:8000/measure-hand"
IMAGE_PATH = "ajfeet.png"

def main():
    with open(IMAGE_PATH, "rb") as f:
        files = {"file": (IMAGE_PATH, f, "image/png")}
        r = requests.post(URL, files=files)

    print("Status:", r.status_code)
    try:
        data = r.json()
    except Exception:
        print("Response is not JSON")
        print(r.text)
        return

    print("Response:")
    print(data)

    # save JSON response for inspection
    try:
        with open("last_hand_response.json.txt", "w", encoding="utf-8") as jf:
            import json
            json.dump(data, jf, indent=2)
        print("Saved JSON response to last_hand_response.json.txt")
    except Exception as e:
        print("Failed to save JSON response:", e)

    if data.get("annotated_image_b64"):
        img_b64 = data["annotated_image_b64"]
        img_bytes = base64.b64decode(img_b64)
        out_path = "annotated.png"
        with open(out_path, "wb") as out:
            out.write(img_bytes)
        print(f"Annotated image saved to {out_path}")


if __name__ == "__main__":
    main()
