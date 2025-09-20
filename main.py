from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import mediapipe as mp
import tempfile

app = FastAPI()

# Paper dimensions in mm (A4 paper: 210 x 297 mm)
PAPER_WIDTH_MM = 210
PAPER_HEIGHT_MM = 297

mp_hands = mp.solutions.hands

# Mediapipe finger landmark indices
FINGERS = {
    "thumb": [1, 2, 3, 4],     # CMC, MCP, IP, TIP
    "index": [5, 6, 7, 8],     # MCP, PIP, DIP, TIP
    "middle": [9, 10, 11, 12], # MCP, PIP, DIP, TIP
    "ring": [13, 14, 15, 16],  # MCP, PIP, DIP, TIP
    "pinky": [17, 18, 19, 20]  # MCP, PIP, DIP, TIP
}

def detect_paper_and_scale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            paper_cnt = approx.reshape(4, 2)
            break
    else:
        raise ValueError("Paper not detected!")

    pts = np.array(sorted(paper_cnt, key=lambda x: (x[1], x[0])))
    top = sorted(pts[:2], key=lambda x: x[0])
    bottom = sorted(pts[2:], key=lambda x: x[0])
    ordered_pts = np.array([top[0], top[1], bottom[1], bottom[0]], dtype="float32")

    width = int(max(
        np.linalg.norm(ordered_pts[0] - ordered_pts[1]),
        np.linalg.norm(ordered_pts[2] - ordered_pts[3])
    ))
    height = int(max(
        np.linalg.norm(ordered_pts[0] - ordered_pts[3]),
        np.linalg.norm(ordered_pts[1] - ordered_pts[2])
    ))

    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(ordered_pts, dst)
    warped = cv2.warpPerspective(image, M, (width, height))

    scale_x = PAPER_WIDTH_MM / width
    scale_y = PAPER_HEIGHT_MM / height
    return warped, (scale_x + scale_y) / 2


def measure_hand(warped_img, scale):
    results_dict = {"fingers": {}, "hand_length_mm": None}
    cv2.imwrite("debug_warped.jpg", warped_img)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        rgb_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_img)

        if not result.multi_hand_landmarks:
            raise ValueError("No hand detected!")

        hand_landmarks = result.multi_hand_landmarks[0]
        h, w, _ = warped_img.shape
        coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

        wrist = coords[0]

        for finger, idxs in FINGERS.items():
            segments = []
            for i in range(len(idxs)):
                if i == 0:
                    start = wrist if finger == "thumb" else coords[idxs[0]]  # thumb base = wrist
                else:
                    start = coords[idxs[i-1]]
                end = coords[idxs[i]]
                dist_px = np.linalg.norm(np.array(end) - np.array(start))
                segments.append(round(dist_px * scale, 2))

            results_dict["fingers"][finger] = {
                "segments_mm": segments,
                "total_length_mm": round(sum(segments), 2)
            }

        results_dict["hand_length_mm"] = results_dict["fingers"]["middle"]["total_length_mm"]

    return results_dict


@app.post("/measure-hand")
async def measure_hand_api(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        img = cv2.imread(tmp_path)
        warped_img, scale = detect_paper_and_scale(img)
        measurements = measure_hand(warped_img, scale)

        return JSONResponse({"success": True, "measurements": measurements})

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})
