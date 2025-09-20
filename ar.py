import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import cv2.aruco as aruco
import mediapipe as mp
import numpy as np

app = FastAPI()

# Finger mapping for returning joint segments (use same indices as MediaPipe)
FINGERS = {
    "thumb": [0, 1, 2, 3, 4],
    "index": [0, 5, 6, 7, 8],
    "middle": [0, 9, 10, 11, 12],
    "ring": [0, 13, 14, 15, 16],
    "pinky": [0, 17, 18, 19, 20]
}


def rectify_with_aruco(image, marker_size_px: int = 300):
    """Detect first ArUco marker and warp the image so the marker maps to a marker_size_px square.
    Returns rectified image and mm_per_px scale (100mm / marker_size_px) when marker found, else
    returns original image and None.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(corners) == 0:
        return image, None

    # Use first detected marker
    marker_corners = corners[0][0].astype(np.float32)
    dst_pts = np.array([
        [0, 0],
        [marker_size_px - 1, 0],
        [marker_size_px - 1, marker_size_px - 1],
        [0, marker_size_px - 1]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(marker_corners, dst_pts)
    h, w = image.shape[:2]
    img_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(img_corners, H)

    xmin, ymin = np.min(warped_corners, axis=0).flatten()
    xmax, ymax = np.max(warped_corners, axis=0).flatten()
    width = int(np.ceil(xmax - xmin))
    height = int(np.ceil(ymax - ymin))

    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])
    H_translated = T @ H
    rectified = cv2.warpPerspective(image, H_translated, (width, height))
    mm_per_px = 100.0 / marker_size_px
    return rectified, mm_per_px


def process_image_for_hand(rectified_img, mm_per_px=None, max_num_hands=1, detection_confidence=0.5):
    """Run MediaPipe hand detector, return landmarks in px and mm (if scale present) and joint segment lengths."""
    mp_hands = mp.solutions.hands
    results_out = {"hands": []}

    with mp_hands.Hands(static_image_mode=True, max_num_hands=max_num_hands, min_detection_confidence=detection_confidence) as hands:
        image_rgb = cv2.cvtColor(rectified_img, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return results_out

        h, w, _ = rectified_img.shape

        for hand_landmarks in results.multi_hand_landmarks:
            landmarks_px = [(float(lm.x * w), float(lm.y * h)) for lm in hand_landmarks.landmark]
            landmarks_mm = None
            if mm_per_px is not None:
                landmarks_mm = [(float(x * mm_per_px), float(y * mm_per_px)) for x, y in landmarks_px]

            # compute segment lengths per finger (in mm when scale available, else px)
            fingers_out = {}
            for fname, ids in FINGERS.items():
                segments = []
                total = 0.0
                for i in range(len(ids) - 1):
                    p1 = landmarks_mm[ids[i]] if landmarks_mm is not None else landmarks_px[ids[i]]
                    p2 = landmarks_mm[ids[i+1]] if landmarks_mm is not None else landmarks_px[ids[i+1]]
                    dist = float(np.linalg.norm(np.array(p2) - np.array(p1)))
                    segments.append(round(dist, 3))
                    total += dist

                fingers_out[fname] = {"segments": segments, "total": round(total, 3)}

            hand_entry = {
                "landmarks_px": [[round(x, 3), round(y, 3)] for x, y in landmarks_px],
                "landmarks_mm": [[round(x, 3), round(y, 3)] for x, y in landmarks_mm] if landmarks_mm is not None else None,
                "fingers": fingers_out
            }
            results_out["hands"].append(hand_entry)

    return results_out


@app.post("/measure-hand")
async def measure_hand_api(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        arr = np.frombuffer(contents, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        rectified, mm_per_px = rectify_with_aruco(image)
        result = process_image_for_hand(rectified, mm_per_px=mm_per_px)

        resp = {"success": True, "scale_mm_per_px": mm_per_px, "result": result}
        return JSONResponse(resp)

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
