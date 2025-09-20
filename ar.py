import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import cv2.aruco as aruco
import mediapipe as mp
import numpy as np


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# During development allow localhost:3000; in production set to your real origin(s)
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # add your production frontend origin(s) here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # or ["*"] for quick dev only
    allow_credentials=True,
    allow_methods=["*"],         # allow POST, OPTIONS, etc.
    allow_headers=["*"],
)


# Finger mapping for returning joint segments (use same indices as MediaPipe)
FINGERS = {
    "thumb": [0, 1, 2, 3, 4],
    "index": [0, 5, 6, 7, 8],
    "middle": [0, 9, 10, 11, 12],
    "ring": [0, 13, 14, 15, 16],
    "pinky": [0, 17, 18, 19, 20]
}


def rectify_with_aruco(image, marker_size_px: int = 300, require_marker: bool = False):
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
        if require_marker:
            # indicate marker was not found
            return None, None, None
        return image, None, None

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

    # transform the original marker corner coordinates into rectified image coords
    # marker_corners is shape (4,2); reshape to (4,1,2) for perspectiveTransform
    marker_corners_reshaped = marker_corners.reshape(-1, 1, 2)
    marker_corners_transformed = cv2.perspectiveTransform(marker_corners_reshaped, H_translated)
    # flatten to (4,2)
    marker_pts = marker_corners_transformed.reshape(-1, 2)

    return rectified, mm_per_px, marker_pts


def process_image_for_hand(rectified_img, mm_per_px=None, marker_pts=None, max_num_hands=1, detection_confidence=0.5):
    """Run MediaPipe hand detector, annotate the image, and return landmarks in px and mm (if scale present), joint segment lengths,
    and the annotated image as a numpy array.
    Returns: {"hands": [...], "annotated_image": <np.ndarray>}
    """
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    results_out = {"hands": [], "annotated_image": None}

    annotated = rectified_img.copy()

    # if marker polygon provided, draw its outline on the annotated image
    if marker_pts is not None:
        try:
            pts = np.array(marker_pts, dtype=int).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            # label
            cx = int(np.mean(marker_pts[:, 0]))
            cy = int(np.mean(marker_pts[:, 1]))
            cv2.putText(annotated, 'ArUco', (cx - 20, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        except Exception:
            # if drawing fails, ignore and continue
            pass

    with mp_hands.Hands(static_image_mode=True, max_num_hands=max_num_hands, min_detection_confidence=detection_confidence) as hands:
        image_rgb = cv2.cvtColor(rectified_img, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            results_out["annotated_image"] = annotated
            return results_out

        h, w, _ = rectified_img.shape

        for hand_landmarks in results.multi_hand_landmarks:
            # draw landmarks & connections
            mp_drawing.draw_landmarks(annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks_px = [(float(lm.x * w), float(lm.y * h)) for lm in hand_landmarks.landmark]
            landmarks_mm = None
            if mm_per_px is not None:
                landmarks_mm = [(float(x * mm_per_px), float(y * mm_per_px)) for x, y in landmarks_px]

            # compute segment lengths per finger (in mm when scale available, else px) and draw labels
            fingers_out = {}
            for fname, ids in FINGERS.items():
                segments = []
                total = 0.0
                for i in range(len(ids) - 1):
                    p1_px = np.array(landmarks_px[ids[i]])
                    p2_px = np.array(landmarks_px[ids[i + 1]])

                    p1 = np.array(landmarks_mm[ids[i]]) if landmarks_mm is not None else p1_px
                    p2 = np.array(landmarks_mm[ids[i + 1]]) if landmarks_mm is not None else p2_px

                    dist = float(np.linalg.norm(p2 - p1))
                    segments.append(round(dist, 3))
                    total += dist

                    # draw segment line and length label on annotated image
                    cv2.line(annotated, tuple(p1_px.astype(int)), tuple(p2_px.astype(int)), (0, 255, 0), 2)
                    mid = ((p1_px + p2_px) / 2).astype(int)
                    text = f"{dist:.1f}{'mm' if landmarks_mm is not None else 'px'}"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    x, y = int(mid[0] - tw / 2), int(mid[1] - th / 2)
                    cv2.rectangle(annotated, (x - 2, y - th - 2), (x + tw + 2, y + 4), (0, 0, 0), -1)
                    cv2.putText(annotated, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

                fingers_out[fname] = {"segments": segments, "total": round(total, 3)}

            hand_entry = {
                "landmarks_px": [[round(x, 3), round(y, 3)] for x, y in landmarks_px],
                "landmarks_mm": [[round(x, 3), round(y, 3)] for x, y in landmarks_mm] if landmarks_mm is not None else None,
                "fingers": fingers_out
            }
            results_out["hands"].append(hand_entry)
        
        # Crop annotated image so the hand takes up most of the image while preserving aspect ratio
        try:
            # collect all landmark points across hands
            all_pts = []
            for hentry in results_out["hands"]:
                if hentry.get("landmarks_px"):
                    all_pts.extend(hentry["landmarks_px"])

            if len(all_pts) > 0:
                arr = np.array(all_pts, dtype=float)
                minx, miny = float(arr[:, 0].min()), float(arr[:, 1].min())
                maxx, maxy = float(arr[:, 0].max()), float(arr[:, 1].max())
                bbox_w = maxx - minx
                bbox_h = maxy - miny

                img_h, img_w = annotated.shape[:2]
                aspect = img_w / img_h if img_h != 0 else 1.0

                # target that bbox occupies ~80% of crop area
                target_frac = 0.8
                desired_w = bbox_w / target_frac if bbox_w > 0 else img_w
                desired_h = bbox_h / target_frac if bbox_h > 0 else img_h

                # enforce aspect ratio: pick crop_w so that crop_h = crop_w / aspect covers desired_h
                crop_w = max(desired_w, desired_h * aspect)
                crop_h = crop_w / aspect

                # clamp to image size
                crop_w = min(crop_w, img_w)
                crop_h = min(crop_h, img_h)

                # center crop around bbox center
                cx = (minx + maxx) / 2.0
                cy = (miny + maxy) / 2.0

                x1f = cx - crop_w / 2.0
                y1f = cy - crop_h / 2.0
                # keep within bounds
                x1f = max(0.0, min(x1f, img_w - crop_w))
                y1f = max(0.0, min(y1f, img_h - crop_h))

                x1 = int(round(x1f))
                y1 = int(round(y1f))
                x2 = int(round(x1 + crop_w))
                y2 = int(round(y1 + crop_h))

                # final bounds safety
                x1 = max(0, min(x1, img_w - 1))
                y1 = max(0, min(y1, img_h - 1))
                x2 = max(0, min(x2, img_w))
                y2 = max(0, min(y2, img_h))

                annotated_cropped = annotated[y1:y2, x1:x2].copy()
                results_out["annotated_image"] = annotated_cropped
            else:
                results_out["annotated_image"] = annotated
        except Exception:
            # on any failure, fall back to full annotated image
            results_out["annotated_image"] = annotated

    return results_out


@app.post("/measure-hand")
async def measure_hand_api(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        arr = np.frombuffer(contents, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Require ArUco detection for reliable scaling
        rectified, mm_per_px, marker_pts = rectify_with_aruco(image, require_marker=True)
        if rectified is None:
            return JSONResponse({"success": False, "error": "ArUco marker not detected"}, status_code=400)

        result = process_image_for_hand(rectified, mm_per_px=mm_per_px, marker_pts=marker_pts)
        if not result.get("hands"):
            return JSONResponse({"success": False, "error": "No hand detected"}, status_code=400)

        # encode annotated image to PNG and base64
        annotated = result.get("annotated_image")

        annotated_b64 = None
        if annotated is not None:
            success, buf = cv2.imencode('.png', annotated)
            if success:
                import base64

                annotated_b64 = base64.b64encode(buf.tobytes()).decode('ascii')

        # remove annotated_image numpy array from result to keep JSON serializable
        result.pop("annotated_image", None)

        resp = {"success": True, "scale_mm_per_px": mm_per_px, "result": result, "annotated_image_b64": annotated_b64}
        return JSONResponse(resp)

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
