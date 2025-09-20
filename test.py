import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load the image
image_path = "test_image.png"  # Change this to your image path
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Convert BGR to RGB for Mediapipe processing
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 1: Detect the paper using contour detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 150)

contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
paper_contour = None
warped = None

if contours:
    # Take the largest contour (likely the paper)
    largest_contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

    if len(approx) >= 4:
        # If we have 4 or more points, we can try to warp it to a rectangle
        pts = approx.reshape(-1, 2)

        # If more than 4 points, keep only the 4 extreme corners using convex hull
        if len(pts) > 4:
            hull = cv2.convexHull(pts)
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
            pts = approx.reshape(-1, 2)

        if len(pts) == 4:
            # Order points: top-left, top-right, bottom-right, bottom-left
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            (tl, tr, br, bl) = rect
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            maxWidth = max(int(widthA), int(widthB))

            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxHeight = max(int(heightA), int(heightB))

            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

            # Draw the outline on original image
            cv2.polylines(image, [approx], isClosed=True, color=(0, 255, 0), thickness=4)

            # Replace image and update dimensions
            image = warped
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            paper_contour = dst  # warped is now the reference frame

            print(f"Warped paper to {maxWidth} x {maxHeight} pixels.")
        else:
            cv2.polylines(image, [approx], isClosed=True, color=(0, 255, 0), thickness=4)
            print("Warning: Could not find exact 4-corner paper shape, using original image.")

# Step 2: Detect hands on the warped (or original) image
with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Example: measure index finger length using paper as reference
            if warped is not None:
                index_tip = hand_landmarks.landmark[8]
                index_mcp = hand_landmarks.landmark[5]
                h, w, _ = image.shape
                tip_px = np.array([index_tip.x * w, index_tip.y * h])
                mcp_px = np.array([index_mcp.x * w, index_mcp.y * h])
                finger_length_px = np.linalg.norm(tip_px - mcp_px)

                # Assume standard A4 height = 297mm
                mm_per_pixel = 297 / h
                finger_length_mm = finger_length_px * mm_per_pixel
                print(f"Index finger length: {finger_length_mm:.2f} mm")

                cv2.putText(
                    image,
                    f"Index: {finger_length_mm:.1f} mm",
                    (int(tip_px[0]), int(tip_px[1] - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
    else:
        print("No hands detected.")

# Display the result
cv2.imshow("Warped Hand + Paper Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
