import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import argparse

class KolamCleaner:
    def __init__(self):
        """
        Initialize the Kolam Cleaner for creating clean white background with blue lines and red dots
        """
        self.line_color = (255, 0, 0)    # Blue color for lines (BGR format)
        self.dot_color = (0, 0, 255)     # Red color for dots (BGR format)
        self.background_color = (255, 255, 255)  # White background
    
    def preprocess_image(self, image):
        """
        Clean and preprocess the input image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Denoise
        denoised = cv.fastNlMeansDenoising(gray, None, h=30, templateWindowSize=7, searchWindowSize=21)
        
        # Threshold to create clean binary image
        _, binary = cv.threshold(denoised, 127, 255, cv.THRESH_BINARY)
        
        # If the pattern is black on white, invert it so pattern is white
        if np.mean(binary) > 127:  # More white pixels than black
            binary = cv.bitwise_not(binary)
        
        # Clean up with morphological operations
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
        clean = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=1)
        clean = cv.morphologyEx(clean, cv.MORPH_CLOSE, kernel, iterations=2)
        
        return clean
    
    def detect_dots(self, binary_image):
        """
        Detect circular dots in the kolam pattern
        """
        # Use HoughCircles to detect circular dots
        circles = cv.HoughCircles(
            binary_image,
            cv.HOUGH_GRADIENT,
            dp=1.2,
            minDist=10,  # Minimum distance between circles
            param1=50,
            param2=15,   # Lower threshold for better detection
            minRadius=2,
            maxRadius=12
        )
        
        dot_locations = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (x, y, r) in circles[0, :]:
                dot_locations.append({
                    "center": (int(x), int(y)),
                    "radius": int(r)
                })
        
        return dot_locations
    
    def create_dot_mask(self, image_shape, dot_locations):
        """
        Create a mask for all detected dots
        """
        dot_mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        for dot in dot_locations:
            cv.circle(dot_mask, dot['center'], dot['radius'] + 1, 255, -1)
        
        return dot_mask
    
    def extract_line_pattern(self, binary_image, dot_mask):
        """
        Extract the line pattern by removing dots from the binary image
        """
        # Remove dots from the binary image to get clean lines
        line_pattern = cv.bitwise_and(binary_image, cv.bitwise_not(dot_mask))
        
        # Clean up the line pattern
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        line_pattern = cv.morphologyEx(line_pattern, cv.MORPH_CLOSE, kernel, iterations=1)
        
        return line_pattern
    
    def create_clean_kolam(self, bw_image_path, output_path="clean_kolam.png"):
        """
        Main function to create clean kolam with white background, blue lines, and red dots
        """
        # Load the image
        image = cv.imread(bw_image_path)
        if image is None:
            print(f"❌ Could not load image: {bw_image_path}")
            return False
        
        print("🧹 Processing kolam image...")
        
        # Preprocess the image
        binary = self.preprocess_image(image)
        
        # Detect dots
        dot_locations = self.detect_dots(binary)
        print(f"🔴 Found {len(dot_locations)} dots")
        
        # Create dot mask
        dot_mask = self.create_dot_mask(binary.shape, dot_locations)
        
        # Extract line pattern (without dots)
        line_pattern = self.extract_line_pattern(binary, dot_mask)
        
        # Create the final clean image
        clean_image = np.full((*binary.shape, 3), 255, dtype=np.uint8)  # White background
        
        # Draw blue lines
        line_coords = np.where(line_pattern > 0)
        if len(line_coords[0]) > 0:
            clean_image[line_coords] = self.line_color
        
        # Draw red dots
        for dot in dot_locations:
            cv.circle(clean_image, dot['center'], dot['radius'], self.dot_color, -1)
        
        # Optional: Make lines thicker for better visibility
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
        line_mask = cv.dilate((clean_image != [255, 255, 255]).any(axis=2).astype(np.uint8) * 255, kernel, iterations=1)
        
        # Redraw the pattern with thicker lines
        clean_image = np.full((*binary.shape, 3), 255, dtype=np.uint8)  # Reset to white
        
        # Apply thicker blue lines
        line_coords = np.where(line_mask > 0)
        clean_image[line_coords] = self.line_color
        
        # Redraw red dots on top
        for dot in dot_locations:
            cv.circle(clean_image, dot['center'], dot['radius'], self.dot_color, -1)
        
        # Save the result
        cv.imwrite(output_path, clean_image)
        print(f"✅ Clean kolam saved to: {output_path}")
        
        # Display results
        self.display_results(image, clean_image)
        
        return True
    
    def display_results(self, original, cleaned):
        """
        Display original and cleaned images side by side
        """
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        if len(original.shape) == 3:
            plt.imshow(cv.cvtColor(original, cv.COLOR_BGR2RGB))
        else:
            plt.imshow(original, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv.cvtColor(cleaned, cv.COLOR_BGR2RGB))
        plt.title('Clean Kolam (White BG, Blue Lines, Red Dots)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Main execution function
    """
    parser = argparse.ArgumentParser(description='Clean Kolam - Create white background with blue lines and red dots')
    parser.add_argument('--input', required=True, help='Path to black & white kolam image')
    parser.add_argument('--output', default='clean_kolam.png', help='Output path for cleaned image')
    
    args = parser.parse_args()
    
    # Initialize cleaner
    cleaner = KolamCleaner()
    
    # Clean the kolam
    success = cleaner.create_clean_kolam(
        bw_image_path=args.input,
        output_path=args.output
    )
    
    if success:
        print(f"\n🎉 Kolam cleaning completed successfully!")
        print(f"📁 Output saved to: {args.output}")
    else:
        print("\n❌ Kolam cleaning failed!")

if __name__ == "__main__":
    # Example usage without command line arguments
    # Uncomment and modify these lines to run directly
    
    # Initialize cleaner
    cleaner = KolamCleaner()
    
    bw_kolam_path = "./black_images/black.png"
    # bw_kolam_path = "final_2BlackKolam.png"
    output_path = "./outputs/color.png"
    
    # Run cleaning process
    print("🧹 Starting Kolam Cleaning Process...")
    success = cleaner.create_clean_kolam(
        bw_image_path=bw_kolam_path,
        output_path=output_path
    )
    
    if success:
        print(f"\n🎉 Your kolam has been cleaned!")
        print(f"📁 Check the output at: {output_path}")
        print("📝 Result: White background, Blue lines, Red dots")
    
    # Uncomment to run main() for command line usage
    # main()