import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import cv2.aruco as aruco
import mediapipe as mp
import numpy as np

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow localhost:3000 for dev (same CORS as ar.py)
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Feature groups (Face Mesh landmark indices). These are common indices but may vary by use-case.
FEATURES = {
    "face_outline": list(range(0, 17)),
    "left_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133],
    "right_eye": [263, 249, 390, 373, 374, 380, 381, 382, 362],
    "nose": [1, 2, 98, 327, 168],
    "mouth": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
}


def rectify_with_aruco(image, marker_size_px: int = 300, require_marker: bool = False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(corners) == 0:
        if require_marker:
            return None, None, None
        return image, None, None

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

    marker_corners_reshaped = marker_corners.reshape(-1, 1, 2)
    marker_corners_transformed = cv2.perspectiveTransform(marker_corners_reshaped, H_translated)
    marker_pts = marker_corners_transformed.reshape(-1, 2)

    return rectified, mm_per_px, marker_pts


def process_image_for_face(rectified_img, mm_per_px=None, marker_pts=None, max_num_faces=1, detection_confidence=0.5):
    mp_face = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    results_out = {"faces": [], "annotated_image": None}
    annotated = rectified_img.copy()

    # draw aruco polygon if present
    if marker_pts is not None:
        try:
            pts = np.array(marker_pts, dtype=int).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            cx = int(np.mean(marker_pts[:, 0]))
            cy = int(np.mean(marker_pts[:, 1]))
            cv2.putText(annotated, 'ArUco', (cx - 20, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        except Exception:
            pass

    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=max_num_faces, refine_landmarks=True, min_detection_confidence=detection_confidence) as face_mesh:
        image_rgb = cv2.cvtColor(rectified_img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            results_out["annotated_image"] = annotated
            return results_out

        h, w, _ = rectified_img.shape
        for face_landmarks in results.multi_face_landmarks:
            # draw mesh
            mp_drawing.draw_landmarks(annotated, face_landmarks, mp_face.FACEMESH_TESSELATION,
                                      mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(0,0,255), thickness=1))

            # collect landmarks
            landmarks_px = [(float(lm.x * w), float(lm.y * h)) for lm in face_landmarks.landmark]
            landmarks_mm = None
            if mm_per_px is not None:
                landmarks_mm = [(float(x * mm_per_px), float(y * mm_per_px)) for x, y in landmarks_px]

            # compute distances for predefined FEATURES
            features_out = {}
            for fname, idxs in FEATURES.items():
                segments = []
                total = 0.0
                for i in range(len(idxs) - 1):
                    a = np.array(landmarks_px[idxs[i]])
                    b = np.array(landmarks_px[idxs[i+1]])
                    dist_px = float(np.linalg.norm(b - a))
                    dist = dist_px * mm_per_px if mm_per_px is not None else dist_px
                    segments.append(round(dist, 3))
                    total += dist
                    # draw segment
                    cv2.line(annotated, tuple(a.astype(int)), tuple(b.astype(int)), (255, 0, 0), 1)
                    mid = ((a + b) / 2).astype(int)
                    unit = 'mm' if mm_per_px is not None else 'px'
                    text = f"{dist:.1f}{unit}"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                    x, y = int(mid[0] - tw / 2), int(mid[1] - th / 2)
                    cv2.rectangle(annotated, (x - 2, y - th - 2), (x + tw + 2, y + 4), (0, 0, 0), -1)
                    cv2.putText(annotated, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA)

                features_out[fname] = {"segments": segments, "total": round(total, 3)}

            face_entry = {
                "landmarks_px": [[round(x,3), round(y,3)] for x,y in landmarks_px],
                "landmarks_mm": [[round(x,3), round(y,3)] for x,y in landmarks_mm] if landmarks_mm is not None else None,
                "features": features_out
            }
            results_out["faces"].append(face_entry)

        # Crop similarly to hand API: crop so face occupies most of frame
        try:
            all_pts = []
            for f in results_out["faces"]:
                if f.get("landmarks_px"):
                    all_pts.extend(f["landmarks_px"])
            if len(all_pts) > 0:
                arr = np.array(all_pts, dtype=float)
                minx, miny = float(arr[:,0].min()), float(arr[:,1].min())
                maxx, maxy = float(arr[:,0].max()), float(arr[:,1].max())
                bbox_w = maxx - minx
                bbox_h = maxy - miny
                img_h, img_w = annotated.shape[:2]
                aspect = img_w / img_h if img_h != 0 else 1.0
                target_frac = 0.8
                desired_w = bbox_w / target_frac if bbox_w > 0 else img_w
                desired_h = bbox_h / target_frac if bbox_h > 0 else img_h
                crop_w = max(desired_w, desired_h * aspect)
                crop_h = crop_w / aspect
                crop_w = min(crop_w, img_w)
                crop_h = min(crop_h, img_h)
                cx = (minx + maxx) / 2.0
                cy = (miny + maxy) / 2.0
                x1f = cx - crop_w / 2.0
                y1f = cy - crop_h / 2.0
                x1f = max(0.0, min(x1f, img_w - crop_w))
                y1f = max(0.0, min(y1f, img_h - crop_h))
                x1 = int(round(x1f)); y1 = int(round(y1f))
                x2 = int(round(x1 + crop_w)); y2 = int(round(y1 + crop_h))
                x1 = max(0, min(x1, img_w - 1)); y1 = max(0, min(y1, img_h - 1))
                x2 = max(0, min(x2, img_w)); y2 = max(0, min(y2, img_h))
                annotated_cropped = annotated[y1:y2, x1:x2].copy()
                results_out["annotated_image"] = annotated_cropped
            else:
                results_out["annotated_image"] = annotated
        except Exception:
            results_out["annotated_image"] = annotated

    return results_out


@app.post("/measure-face")
async def measure_face_api(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        arr = np.frombuffer(contents, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        rectified, mm_per_px, marker_pts = rectify_with_aruco(image, require_marker=True)
        if rectified is None:
            return JSONResponse({"success": False, "error": "ArUco marker not detected"}, status_code=400)

        result = process_image_for_face(rectified, mm_per_px=mm_per_px, marker_pts=marker_pts)
        if not result.get("faces"):
            return JSONResponse({"success": False, "error": "No face detected"}, status_code=400)

        annotated = result.get("annotated_image")
        annotated_b64 = None
        if annotated is not None:
            success, buf = cv2.imencode('.png', annotated)
            if success:
                import base64
                annotated_b64 = base64.b64encode(buf.tobytes()).decode('ascii')

        result.pop("annotated_image", None)
        resp = {"success": True, "scale_mm_per_px": mm_per_px, "result": result, "annotated_image_b64": annotated_b64}
        return JSONResponse(resp)

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
