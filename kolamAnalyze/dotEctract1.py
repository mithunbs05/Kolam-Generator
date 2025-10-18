import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import math

class ImprovedKolamDotDetector:
   
    def __init__(self):
        self.detected_dots = []
        self.grid_spacing = None

    def preprocess_image(self, image):
        """
        Preprocess the kolam image for better dot detection
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(blurred)
        
        return enhanced

    def filter_edge_dots(self, dots, image_shape, margin_percent=0.05):
        """
        Remove dots that are too close to the image border using percentage-based margin
        """
        h, w = image_shape[:2]
        margin_x = int(w * margin_percent)
        margin_y = int(h * margin_percent)
        
        filtered = []
        for x, y, r in dots:
            if margin_x < x < (w - margin_x) and margin_y < y < (h - margin_y):
                filtered.append((x, y, r))
        
        print(f"Edge filtering: {len(dots)} -> {len(filtered)} dots (removed {len(dots) - len(filtered)} edge dots)")
        return filtered

    def remove_duplicates_improved(self, dots, min_distance_factor=2.0):
        """
        Improved duplicate removal with adaptive distance threshold
        """
        if len(dots) == 0:
            return []
        
        # Estimate average radius for adaptive thresholding
        avg_radius = np.mean([r for _, _, r in dots])
        min_distance = int(avg_radius * min_distance_factor)
        
        print(f"Using min_distance = {min_distance} (avg_radius = {avg_radius:.1f})")
            
        # Convert to array for easier manipulation
        dots_array = np.array([(x, y) for x, y, r in dots])
        
        # Use DBSCAN clustering to group nearby points
        clustering = DBSCAN(eps=min_distance, min_samples=1).fit(dots_array)
        labels = clustering.labels_
        
        unique_dots = []
        for label in set(labels):
            if label == -1:  # Noise points
                continue
            cluster_mask = labels == label
            cluster_dots = [dots[i] for i in range(len(dots)) if cluster_mask[i]]
            
            # Take the dot with median coordinates in each cluster
            if len(cluster_dots) > 0:
                x_coords = [dot[0] for dot in cluster_dots]
                y_coords = [dot[1] for dot in cluster_dots]
                radii = [dot[2] for dot in cluster_dots]
                
                median_x = int(np.median(x_coords))
                median_y = int(np.median(y_coords))
                median_r = int(np.median(radii))
                
                unique_dots.append((median_x, median_y, median_r))
        
        print(f"Duplicate removal: {len(dots)} -> {len(unique_dots)} dots (removed {len(dots) - len(unique_dots)} duplicates)")
        return unique_dots

    def detect_dots_hough_circles(self, image, min_radius=3, max_radius=25):
        """
        Detect dots using Hough Circle Transform with improved parameters
        """
        preprocessed = self.preprocess_image(image)
        
        # Apply Hough Circle Transform with more conservative parameters
        circles = cv2.HoughCircles(
            preprocessed,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,  # Increased minimum distance between circle centers
            param1=50,   # Upper threshold for edge detection
            param2=35,   # Increased accumulator threshold for center detection
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        dots = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                dots.append((x, y, r))
        
        print(f"Hough circles detected: {len(dots)} dots")
        return dots

    def detect_dots_blob_detection(self, image):
        """
        Detect dots using blob detection with stricter parameters
        """
        preprocessed = self.preprocess_image(image)
        
        # Setup SimpleBlobDetector parameters with stricter filtering
        params = cv2.SimpleBlobDetector_Params()
        
        # Filter by Area
        params.filterByArea = True
        params.minArea = 20  # Increased minimum area
        params.maxArea = 250  # Decreased maximum area
        
        # Filter by Circularity (more strict)
        params.filterByCircularity = True
        params.minCircularity = 0.75  # Increased from 0.6
        
        # Filter by Convexity (more strict)
        params.filterByConvexity = True
        params.minConvexity = 0.8  # Increased from 0.7
        
        # Filter by Inertia (roundness) - more strict
        params.filterByInertia = True
        params.minInertiaRatio = 0.6  # Increased from 0.5
        
        # Create detector
        detector = cv2.SimpleBlobDetector_create(params)
        
        # Detect blobs
        keypoints = detector.detect(preprocessed)
        
        dots = []
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            r = int(kp.size / 2)
            dots.append((x, y, r))
        
        print(f"Blob detection found: {len(dots)} dots")
        return dots

    def detect_dots_contour_based(self, image):
        """
        Detect dots using contour analysis with improved filtering
        """
        preprocessed = self.preprocess_image(image)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            preprocessed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        dots = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 20 or area > 250:  # More restrictive area filtering
                continue
                
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            
            if circularity > 0.7:  # More strict circularity threshold
                # Get centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Estimate radius from area
                    radius = int(math.sqrt(area / math.pi))
                    dots.append((cx, cy, radius))
        
        print(f"Contour detection found: {len(dots)} dots")
        return dots

    def estimate_grid_spacing(self, dots):
        """
        Estimate the grid spacing from detected dots
        """
        if len(dots) < 4:
            return None
            
        # Calculate distances between all pairs of dots
        points = [(x, y) for x, y, r in dots]
        distances = []
        
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = math.sqrt((points[i][0] - points[j][0])**2 + 
                               (points[i][1] - points[j][1])**2)
                distances.append(dist)
        
        # Find the most common distance (likely grid spacing)
        distances.sort()
        
        # Use histogram to find the most frequent distance
        hist, bins = np.histogram(distances, bins=30)
        max_freq_idx = np.argmax(hist)
        estimated_spacing = (bins[max_freq_idx] + bins[max_freq_idx + 1]) / 2
        
        return estimated_spacing

    def conservative_hybrid_detection(self, image):
        """
        Conservative hybrid detection that prioritizes accuracy over recall
        """
        print("\n=== Conservative Hybrid Detection ===")
        
        # Get results from most reliable methods
        hough_dots = self.detect_dots_hough_circles(image)
        blob_dots = self.detect_dots_blob_detection(image)
        
        # Apply edge filtering to each method separately
        hough_filtered = self.filter_edge_dots(hough_dots, image.shape, margin_percent=0.08)
        blob_filtered = self.filter_edge_dots(blob_dots, image.shape, margin_percent=0.08)
        
        # Combine results
        all_dots = hough_filtered + blob_filtered
        print(f"Combined detections: {len(all_dots)} dots")
        
        # Remove duplicates with adaptive threshold
        unique_dots = self.remove_duplicates_improved(all_dots, min_distance_factor=1.8)
        
        # Estimate grid spacing
        self.grid_spacing = self.estimate_grid_spacing(unique_dots)
        
        return unique_dots

    def count_dots(self, image_path, method='conservative_hybrid', visualize=True):
        """
        Main function to count dots in a kolam image
        
        Parameters:
        - image_path: path to the kolam image
        - method: detection method ('hough', 'blob', 'contour', 'conservative_hybrid')
        - visualize: whether to show the result
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return 0
        
        print(f"Image shape: {image.shape}")
        
        # Choose detection method
        if method == 'hough':
            dots = self.filter_edge_dots(self.detect_dots_hough_circles(image), image.shape)
            dots = self.remove_duplicates_improved(dots)
        elif method == 'blob':
            dots = self.filter_edge_dots(self.detect_dots_blob_detection(image), image.shape)
            dots = self.remove_duplicates_improved(dots)
        elif method == 'contour':
            dots = self.filter_edge_dots(self.detect_dots_contour_based(image), image.shape)
            dots = self.remove_duplicates_improved(dots)
        elif method == 'conservative_hybrid':
            dots = self.conservative_hybrid_detection(image)
        else:
            print(f"Unknown method: {method}")
            return 0
        
        self.detected_dots = dots
        dot_count = len(dots)
        
        print(f"\n=== FINAL RESULT ===")
        print(f"Total dots detected: {dot_count}")
        
        if visualize:
            self.visualize_results(image, dots, dot_count)
        
        return dot_count

    def visualize_results(self, image, dots, count):
        """
        Visualize the detected dots on the original image
        """
        # Create a copy for visualization
        result_image = image.copy()
        
        # Draw detected dots
        for i, (x, y, r) in enumerate(dots):
            # Draw circle
            cv2.circle(result_image, (x, y), r, (0, 255, 0), 2)
            # Draw center point
            cv2.circle(result_image, (x, y), 2, (0, 0, 255), -1)
            # Add number label
            cv2.putText(result_image, str(i+1), (x+r+5, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Add count text
        cv2.putText(result_image, f'Total Dots: {count}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        if self.grid_spacing:
            cv2.putText(result_image, f'Grid Spacing: {self.grid_spacing:.1f}px', 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display results
        plt.figure(figsize=(15, 10))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Detected Dots (Count: {count})')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    def save_results(self, image_path, output_path=None):
        """
        Save the detection results to a file
        """
        if output_path is None:
            output_path = image_path.replace('.', '_results.')
        
        # Load original image
        image = cv2.imread(image_path)
        result_image = image.copy()
        
        # Draw detected dots
        for i, (x, y, r) in enumerate(self.detected_dots):
            cv2.circle(result_image, (x, y), r, (0, 255, 0), 2)
            cv2.circle(result_image, (x, y), 2, (0, 0, 255), -1)
            cv2.putText(result_image, str(i+1), (x+r+5, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Add count text
        cv2.putText(result_image, f'Total Dots: {len(self.detected_dots)}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Save result
        cv2.imwrite(output_path, result_image)
        print(f"Results saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    # Initialize improved detector
    detector = ImprovedKolamDotDetector()

    image_path = "final_3colorImage1.png"
    # image_path = "./outputs/color.png"
    
    print("Testing improved detection methods:")
    print("=" * 50)
    
    methods = ['hough', 'blob', 'contour', 'conservative_hybrid']
    
    for method in methods:
        try:
            print(f"\n--- Testing {method.upper()} method ---")
            count = detector.count_dots(image_path, method=method, visualize=False)
            print(f"Final count for {method}: {count} dots")
        except Exception as e:
            print(f"{method.capitalize()} method: Error - {str(e)}")
    
    print("\n" + "=" * 50)
    print("Using conservative hybrid method with visualization:")
    
    # Use conservative hybrid method for best accuracy
    final_count = detector.count_dots(image_path, method='conservative_hybrid', visualize=True)
    
    # Save results
    detector.save_results(image_path, "./outputs/result.jpg")
    
    print(f"\nFINAL DOT COUNT: {final_count}")
    if detector.grid_spacing:
        print(f"Estimated grid spacing: {detector.grid_spacing:.2f} pixels")