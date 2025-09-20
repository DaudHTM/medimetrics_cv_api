import cv2
import numpy as np
from pathlib import Path

def find_paper_contour(image, show_debug=False):
    """
    Find the largest rectangular contour in the image (assumed to be the paper)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Try multiple preprocessing approaches
    methods = []
    
    # Method 1: Standard approach
    blurred1 = cv2.GaussianBlur(gray, (5, 5), 0)
    edges1 = cv2.Canny(blurred1, 50, 150, apertureSize=3)
    methods.append(("Standard", edges1))
    
    # Method 2: More aggressive edge detection
    blurred2 = cv2.GaussianBlur(gray, (3, 3), 0)
    edges2 = cv2.Canny(blurred2, 30, 100, apertureSize=3)
    methods.append(("Aggressive", edges2))
    
    # Method 3: Adaptive threshold
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    edges3 = cv2.Canny(adaptive, 50, 150)
    methods.append(("Adaptive", edges3))
    
    # Method 4: Morphological operations first
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    edges4 = cv2.Canny(morph, 50, 150)
    methods.append(("Morphological", edges4))
    
    best_contour = None
    best_method = ""
    
    for method_name, edges in methods:
        # Apply morphological operations to close gaps
        kernel = np.ones((3, 3), np.uint8)
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges_dilated = cv2.dilate(edges_closed, kernel, iterations=1)
        
        if show_debug:
            cv2.imshow(f"Edges - {method_name}", cv2.resize(edges_dilated, (400, 300)))
        
        # Find contours
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            continue
            
        # Sort contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        print(f"{method_name} method found {len(contours)} contours")
        
        # Look for suitable contours
        for i, contour in enumerate(contours[:10]):  # Check top 10 contours
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if area < 1000:  # Skip very small contours
                continue
                
            # Try different epsilon values for approximation
            for epsilon_factor in [0.01, 0.02, 0.03, 0.04, 0.05]:
                epsilon = epsilon_factor * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                print(f"  Contour {i}: Area={area:.0f}, Points={len(approx)}, Epsilon={epsilon_factor}")
                
                # Accept quadrilateral or close to it (3-6 points)
                if 3 <= len(approx) <= 6 and area > 5000:
                    if len(approx) != 4:
                        # If not exactly 4 points, try to fit a bounding rectangle
                        rect = cv2.minAreaRect(contour)
                        box = cv2.boxPoints(rect)
                        approx = np.int0(box)
                    
                    best_contour = approx
                    best_method = method_name
                    print(f"  -> Selected this contour from {method_name} method")
                    break
            
            if best_contour is not None:
                break
        
        if best_contour is not None:
            break
    
    if show_debug:
        cv2.waitKey(0)
    
    print(f"Best detection method: {best_method}" if best_contour is not None else "No suitable contour found")
    return best_contour

def order_points(pts):
    """
    Order points in the order: top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sum and difference to find corners
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    
    return rect

def four_point_transform(image, pts):
    """
    Apply perspective transformation to get a top-down view of the paper
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Calculate width and height of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Define destination points for the transformation
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Calculate perspective transformation matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def crop_paper_from_image(image_path, output_path=None, show_debug=False):
    """
    Main function to crop paper from image
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    print(f"Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    original = image.copy()
    
    # Find paper contour
    paper_contour = find_paper_contour(image, show_debug)
    
    if paper_contour is None:
        print("Error: Could not detect paper in the image")
        print("Try setting show_debug=True to see the edge detection results")
        print("Make sure the paper has clear, contrasting edges against the background")
        
        # Show original image for manual inspection
        cv2.imshow("Original Image - No Paper Detected", cv2.resize(original, (800, 600)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None
    
    if show_debug:
        # Draw the detected contour on the original image
        debug_img = original.copy()
        cv2.drawContours(debug_img, [paper_contour], -1, (0, 255, 0), 3)
        # Draw corner points
        for i, point in enumerate(paper_contour):
            cv2.circle(debug_img, tuple(point[0]), 10, (0, 0, 255), -1)
            cv2.putText(debug_img, str(i), tuple(point[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Detected Paper", cv2.resize(debug_img, (800, 600)))
        cv2.waitKey(0)
    
    # Apply perspective transformation
    paper_contour = paper_contour.reshape(4, 2)
    cropped_paper = four_point_transform(original, paper_contour)
    
    # Save the result if output path is provided
    if output_path:
        cv2.imwrite(output_path, cropped_paper)
        print(f"Cropped image saved to: {output_path}")
    
    # Display results
    cv2.imshow("Original Image", cv2.resize(original, (800, 600)))
    cv2.imshow("Cropped Paper", cv2.resize(cropped_paper, (800, 600)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return cropped_paper

def main():
    # Set your image path here
    image_path = "test_image.png"  # Change this to your image path
    
    # Optional settings
    show_debug = True  # Set to True to see detection visualization and debugging info
    
    # Verify input file exists
    if not Path(image_path).exists():
        print(f"Error: Image file {image_path} not found")
        return
    
    # Set output path
    input_path = Path(image_path)
    output_path = input_path.parent / f"{input_path.stem}_cropped{input_path.suffix}"
    
    # Process the image
    crop_paper_from_image(image_path, str(output_path), show_debug)

if __name__ == "__main__":
    main()