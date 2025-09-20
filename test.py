import cv2
import numpy as np
import os
from typing import List, Tuple, Optional

class DocumentScanner:
    def __init__(self):
        self.min_contour_area = 5000
        self.epsilon_factor = 0.02
        
    def resize_image(self, image: np.ndarray, width: int = 800) -> np.ndarray:
        """Resize image while maintaining aspect ratio"""
        height, orig_width = image.shape[:2]
        ratio = width / orig_width
        new_height = int(height * ratio)
        return cv2.resize(image, (width, new_height))
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better edge detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply edge detection
        edges = cv2.Canny(blurred, 75, 200)
        
        # Dilate and erode to close gaps
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        return edges
    
    def find_document_contour(self, edges: np.ndarray) -> Optional[np.ndarray]:
        """Find the largest rectangular contour (document)"""
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours:
            # Skip small contours
            if cv2.contourArea(contour) < self.min_contour_area:
                continue
                
            # Approximate contour to polygon
            epsilon = self.epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # If we found a 4-sided polygon, it's likely our document
            if len(approx) == 4:
                return approx
                
        return None
    
    def order_points(self, points: np.ndarray) -> np.ndarray:
        """Order points in clockwise order: top-left, top-right, bottom-right, bottom-left"""
        points = points.reshape(4, 2)
        ordered = np.zeros((4, 2), dtype=np.float32)
        
        # Sum and difference to find corners
        s = points.sum(axis=1)
        diff = np.diff(points, axis=1)
        
        # Top-left has smallest sum, bottom-right has largest sum
        ordered[0] = points[np.argmin(s)]  # top-left
        ordered[2] = points[np.argmax(s)]  # bottom-right
        
        # Top-right has smallest difference, bottom-left has largest difference
        ordered[1] = points[np.argmin(diff)]  # top-right
        ordered[3] = points[np.argmax(diff)]  # bottom-left
        
        return ordered
    
    def get_perspective_transform(self, src_points: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate perspective transform matrix"""
        # Order the source points
        src = self.order_points(src_points)
        
        # Define destination points for A4 ratio or custom dimensions
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        transform_matrix = cv2.getPerspectiveTransform(src, dst)
        
        return transform_matrix, dst
    
    def apply_perspective_correction(self, image: np.ndarray, contour: np.ndarray, output_width: int = 800, output_height: int = 1000) -> np.ndarray:
        """Apply perspective correction to extract document"""
        # Get perspective transform
        transform_matrix, _ = self.get_perspective_transform(contour, output_width, output_height)
        
        # Apply perspective transformation
        warped = cv2.warpPerspective(image, transform_matrix, (output_width, output_height))
        
        return warped
    
    def enhance_document(self, image: np.ndarray) -> np.ndarray:
        """Enhance the scanned document"""
        # Convert to grayscale if it's color
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive thresholding for better text contrast
        enhanced = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 10
        )
        
        # Optional: Apply morphological operations to clean up noise
        kernel = np.ones((2, 2), np.uint8)
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        return enhanced
    
    def scan_document(self, image_path: str, output_path: Optional[str] = None, 
                     show_steps: bool = False, enhance: bool = True) -> np.ndarray:
        """Main function to scan a document"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Store original for later use
        original = image.copy()
        
        # Resize for faster processing
        image = self.resize_image(image)
        ratio = original.shape[1] / image.shape[1]
        
        # Preprocess image
        edges = self.preprocess_image(image)
        
        if show_steps:
            cv2.imshow("Original", self.resize_image(original, 600))
            cv2.imshow("Edges", edges)
        
        # Find document contour
        doc_contour = self.find_document_contour(edges)
        
        if doc_contour is None:
            print("Warning: Could not find document contour. Using full image.")
            scanned = original
        else:
            # Scale contour back to original image size
            doc_contour = doc_contour.astype(np.float32) * ratio
            
            # Show detected contour
            if show_steps:
                contour_img = original.copy()
                cv2.drawContours(contour_img, [doc_contour.astype(np.int32)], -1, (0, 255, 0), 3)
                cv2.imshow("Detected Document", self.resize_image(contour_img, 600))
            
            # Apply perspective correction
            scanned = self.apply_perspective_correction(original, doc_contour)
        
        # Enhance the document
        if enhance:
            scanned = self.enhance_document(scanned)
        
        # Save result
        if output_path:
            cv2.imwrite(output_path, scanned)
            print(f"Scanned document saved to: {output_path}")
        
        if show_steps:
            cv2.imshow("Scanned Document", self.resize_image(scanned, 600))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return scanned

def main():
    # Find image files in current directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = [f for f in os.listdir('.') if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    if not image_files:
        print("No image files found in current directory!")
        print("Please place an image file (.jpg, .png, etc.) in the same directory.")
        return
    
    print("Available images:")
    for i, img in enumerate(image_files, 1):
        print(f"{i}. {img}")
    
    # Get user choice
    try:
        choice = input("\nEnter image number or filename: ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(image_files):
            selected_image = image_files[int(choice) - 1]
        elif choice in image_files:
            selected_image = choice
        else:
            print(f"Invalid choice. Using first image: {image_files[0]}")
            selected_image = image_files[0]
            
    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
        return
    
    # Create scanner and process image
    scanner = DocumentScanner()
    
    try:
        print(f"\nProcessing: {selected_image}")
        output_name = f"scanned_{selected_image}"
        
        result = scanner.scan_document(
            selected_image, 
            output_name, 
            show_steps=True,
            enhance=True
        )
        print("Document scanning completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()