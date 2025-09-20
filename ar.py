import cv2
import cv2.aruco as aruco
import mediapipe as mp
import numpy as np

# --- Load image ---
image_path = "test1.png"
image = cv2.imread(image_path)

if image is None:
    print("Could not load image!")
    exit()

# --- Convert to grayscale for ArUco ---
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --- ArUco Setup ---
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# --- Detect ArUco Markers ---
corners, ids, rejected = detector.detectMarkers(gray)

mm_per_px = None

if ids is not None:
    print(f"Detected ArUco IDs: {ids.flatten()}")
    aruco.drawDetectedMarkers(image, corners, ids)

    marker_corners = corners[0][0]
    marker_size_px = 300  # warp marker to 300px

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

    [xmin, ymin] = np.min(warped_corners, axis=0).flatten()
    [xmax, ymax] = np.max(warped_corners, axis=0).flatten()

    width = int(np.ceil(xmax - xmin))
    height = int(np.ceil(ymax - ymin))

    T = np.array([[1, 0, -xmin],
                  [0, 1, -ymin],
                  [0, 0, 1]])
    H_translated = T @ H

    rectified = cv2.warpPerspective(image, H_translated, (width, height))

    mm_per_px = 100.0 / marker_size_px
else:
    print("No ArUco markers detected.")
    rectified = image.copy()

# --- MediaPipe Hand Detection ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
) as hands:

    image_rgb = cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks and mm_per_px:
        h, w, _ = rectified.shape

        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(rectified, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks_px = np.array([
                [lm.x * w, lm.y * h] for lm in hand_landmarks.landmark
            ])
            landmarks_mm = landmarks_px * mm_per_px

            # Define finger landmark indices
            fingers = {
                "Thumb":  [0, 1, 2, 3, 4],
                "Index":  [0, 5, 6, 7, 8],
                "Middle": [0, 9, 10, 11, 12],
                "Ring":   [0, 13, 14, 15, 16],
                "Pinky":  [0, 17, 18, 19, 20],
            }

            # Draw and label each segment
            for finger_name, ids in fingers.items():
                for i in range(len(ids)-1):
                    p1_px = landmarks_px[ids[i]]
                    p2_px = landmarks_px[ids[i+1]]
                    p1_mm = landmarks_mm[ids[i]]
                    p2_mm = landmarks_mm[ids[i+1]]

                    length_mm = np.linalg.norm(p2_mm - p1_mm)

                    # Draw line
                    cv2.line(rectified,
                             tuple(np.int32(p1_px)),
                             tuple(np.int32(p2_px)),
                             (0, 255, 0), 2)

                    # Draw label near midpoint
                    mid_pt = (p1_px + p2_px) / 2
                    text = f"{length_mm:.1f}mm"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    # Background rectangle for visibility
                    x, y = int(mid_pt[0] - tw/2), int(mid_pt[1] - th/2)
                    cv2.rectangle(rectified, (x-2, y-th-2), (x+tw+2, y+4), (0, 0, 0), -1)

                    # Put text
                    cv2.putText(rectified, text, (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        print(f"\nScale: {mm_per_px:.3f} mm/px")
    else:
        print("No hands detected or scale not available.")

# --- Show Final Result ---
cv2.imshow("Hand Measurement in mm", rectified)
cv2.waitKey(0)
cv2.destroyAllWindows()
