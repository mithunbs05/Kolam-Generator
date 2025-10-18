import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc
import math
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN
import json

class KolamAnalyzer:
    def __init__(self):
        self.dots = []
        self.contours = []
        self.grid_size = (0, 0)
        self.is_symmetric = False
        self.design_principles = {}
        
    def load_image(self, image_path):
        """Load and preprocess the Kolam image"""
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("Could not load image")
        
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.gray.shape
        return self.image
    
    def detect_dots(self, min_radius=3, max_radius=15):
        """Detect dots in the Kolam design using HoughCircles"""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(self.gray, (9, 9), 2)
        
        # Use HoughCircles to detect circular dots
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            self.dots = [(x, y, r) for x, y, r in circles]
        else:
            self.dots = []
        
        return self.dots
    
    def detect_contours(self):
        """Detect contours (lines/curves) in the Kolam design"""
        # Create binary image
        _, binary = cv2.threshold(self.gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Remove dots from binary image to isolate lines
        for x, y, r in self.dots:
            cv2.circle(binary, (x, y), r + 2, 0, -1)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and length
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if area > 50 and perimeter > 20:  # Adjust thresholds as needed
                filtered_contours.append(contour)
        
        self.contours = filtered_contours
        return self.contours
    
    def determine_grid_structure(self):
        """Determine the grid structure from dot positions"""
        if not self.dots:
            return (0, 0)
        
        # Extract x and y coordinates
        x_coords = [dot[0] for dot in self.dots]
        y_coords = [dot[1] for dot in self.dots]
        
        # Use clustering to find grid positions
        if len(set(x_coords)) > 1 and len(set(y_coords)) > 1:
            # Cluster x coordinates
            x_clusters = DBSCAN(eps=20, min_samples=1).fit([[x] for x in x_coords])
            x_unique = len(set(x_clusters.labels_))
            
            # Cluster y coordinates
            y_clusters = DBSCAN(eps=20, min_samples=1).fit([[y] for y in y_coords])
            y_unique = len(set(y_clusters.labels_))
            
            self.grid_size = (x_unique, y_unique)
        else:
            self.grid_size = (len(set(x_coords)), len(set(y_coords)))
        
        return self.grid_size
    
    def check_symmetry(self):
        """Check if the design has symmetrical properties"""
        if not self.dots:
            return False
        
        # Get center of the design
        center_x = sum(dot[0] for dot in self.dots) / len(self.dots)
        center_y = sum(dot[1] for dot in self.dots) / len(self.dots)
        
        # Check horizontal symmetry
        symmetric_pairs = 0
        total_pairs = 0
        
        for i, (x1, y1, r1) in enumerate(self.dots):
            for j, (x2, y2, r2) in enumerate(self.dots[i+1:], i+1):
                total_pairs += 1
                
                # Check if points are symmetric about center
                expected_x = 2 * center_x - x1
                expected_y = 2 * center_y - y1
                
                if abs(expected_x - x2) < 10 and abs(expected_y - y2) < 10:
                    symmetric_pairs += 1
        
        self.is_symmetric = symmetric_pairs / max(total_pairs, 1) > 0.5
        return self.is_symmetric
    
    def analyze_connections(self):
        """Analyze how contours connect or interact with dots"""
        connections = []
        
        for i, contour in enumerate(self.contours):
            contour_connections = {
                'contour_id': i,
                'connected_dots': [],
                'path_type': 'unknown',
                'curves': []
            }
            
            # Check which dots this contour passes near or around
            for j, (dot_x, dot_y, dot_r) in enumerate(self.dots):
                # Calculate minimum distance from contour to dot
                min_dist = cv2.pointPolygonTest(contour, (dot_x, dot_y), True)
                
                if abs(min_dist) <= dot_r + 5:  # Within dot radius + tolerance
                    contour_connections['connected_dots'].append({
                        'dot_id': j,
                        'dot_pos': (dot_x, dot_y),
                        'interaction_type': 'around' if min_dist < 0 else 'near'
                    })
            
            # Analyze contour shape
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) < 4:
                contour_connections['path_type'] = 'line'
            elif len(approx) < 8:
                contour_connections['path_type'] = 'polygon'
            else:
                contour_connections['path_type'] = 'curve'
            
            connections.append(contour_connections)
        
        return connections
    
    def extract_design_principles(self):
        """Extract key design principles from the analysis"""
        connections = self.analyze_connections()
        
        self.design_principles = {
            'dot_count': len(self.dots),
            'contour_count': len(self.contours),
            'grid_size': self.grid_size,
            'is_symmetric': self.is_symmetric,
            'dot_positions': [(x, y) for x, y, r in self.dots],
            'connections': connections,
            'pattern_type': self.classify_pattern()
        }
        
        return self.design_principles
    
    def classify_pattern(self):
        """Classify the type of Kolam pattern"""
        dot_count = len(self.dots)
        contour_count = len(self.contours)
        
        if dot_count == 0:
            return "free_form"
        elif contour_count > dot_count:
            return "complex_interwoven"
        elif contour_count == 1:
            return "single_loop"
        elif self.is_symmetric:
            return "symmetric_pattern"
        else:
            return "asymmetric_pattern"
    
    def visualize_analysis(self, save_path=None):
        """Visualize the analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Kolam')
        axes[0, 0].axis('off')
        
        # Detected dots
        axes[0, 1].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        for x, y, r in self.dots:
            circle = Circle((x, y), r, fill=False, color='red', linewidth=2)
            axes[0, 1].add_patch(circle)
            axes[0, 1].text(x, y-r-5, f'({x},{y})', ha='center', fontsize=8, color='red')
        axes[0, 1].set_title(f'Detected Dots ({len(self.dots)})')
        axes[0, 1].axis('off')
        
        # Detected contours
        axes[1, 0].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        for i, contour in enumerate(self.contours):
            # Draw contour
            cv2.drawContours(self.image.copy(), [contour], -1, (0, 255, 0), 2)
        axes[1, 0].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title(f'Detected Contours ({len(self.contours)})')
        axes[1, 0].axis('off')
        
        # Analysis summary
        axes[1, 1].axis('off')
        summary_text = f"""
Analysis Results:
• Dot Count: {len(self.dots)}
• Contour Count: {len(self.contours)}
• Grid Size: {self.grid_size[0]} × {self.grid_size[1]}
• Symmetric: {'Yes' if self.is_symmetric else 'No'}
• Pattern Type: {self.classify_pattern()}

Dot Positions:
{chr(10).join([f'• Dot {i+1}: ({x}, {y})' for i, (x, y, r) in enumerate(self.dots)])}
        """
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                        verticalalignment='top', fontsize=10, fontfamily='monospace')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class KolamRecreator:
    def __init__(self, design_principles):
        self.principles = design_principles
        self.canvas_size = (400, 400)
        
    def recreate_kolam(self, output_path=None):
        """Recreate the Kolam based on extracted design principles"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_xlim(0, self.canvas_size[0])
        ax.set_ylim(0, self.canvas_size[1])
        ax.set_aspect('equal')
        
        # Scale dot positions to canvas
        if self.principles['dot_positions']:
            # Find bounds of original dots
            orig_x = [pos[0] for pos in self.principles['dot_positions']]
            orig_y = [pos[1] for pos in self.principles['dot_positions']]
            
            min_x, max_x = min(orig_x), max(orig_x)
            min_y, max_y = min(orig_y), max(orig_y)
            
            # Scale to canvas with padding
            padding = 50
            scale_x = (self.canvas_size[0] - 2 * padding) / max(max_x - min_x, 1)
            scale_y = (self.canvas_size[1] - 2 * padding) / max(max_y - min_y, 1)
            scale = min(scale_x, scale_y)
            
            # Draw dots
            scaled_dots = []
            for x, y in self.principles['dot_positions']:
                scaled_x = padding + (x - min_x) * scale
                scaled_y = padding + (y - min_y) * scale
                scaled_dots.append((scaled_x, scaled_y))
                
                circle = Circle((scaled_x, scaled_y), 5, fill=True, color='black')
                ax.add_patch(circle)
            
            # Draw connections based on pattern type
            self.draw_pattern_connections(ax, scaled_dots)
        
        ax.set_title(f'Recreated Kolam - {self.principles["pattern_type"]}')
        ax.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def draw_pattern_connections(self, ax, dots):
        """Draw connections based on the pattern type"""
        pattern_type = self.principles.get('pattern_type', 'single_loop')
        
        if pattern_type == 'single_loop' and len(dots) >= 2:
            self.draw_single_loop(ax, dots)
        elif pattern_type == 'symmetric_pattern':
            self.draw_symmetric_pattern(ax, dots)
        elif len(dots) >= 2:
            self.draw_basic_connections(ax, dots)
    
    def draw_single_loop(self, ax, dots):
        """Draw a single continuous loop around/through dots"""
        if len(dots) < 2:
            return
            
        # Create a smooth path around dots
        for i in range(len(dots)):
            start_dot = dots[i]
            end_dot = dots[(i + 1) % len(dots)]
            
            # Draw curved connection
            mid_x = (start_dot[0] + end_dot[0]) / 2
            mid_y = (start_dot[1] + end_dot[1]) / 2
            
            # Add curvature
            offset = 20
            if i % 2 == 0:
                control_x = mid_x + offset
                control_y = mid_y + offset
            else:
                control_x = mid_x - offset
                control_y = mid_y - offset
            
            # Draw Bezier curve (approximated with arc)
            self.draw_curved_line(ax, start_dot, end_dot, (control_x, control_y))
    
    def draw_symmetric_pattern(self, ax, dots):
        """Draw symmetric patterns"""
        if len(dots) < 2:
            return
            
        center_x = sum(dot[0] for dot in dots) / len(dots)
        center_y = sum(dot[1] for dot in dots) / len(dots)
        
        # Draw radial connections from center
        for dot in dots:
            self.draw_curved_line(ax, (center_x, center_y), dot)
    
    def draw_basic_connections(self, ax, dots):
        """Draw basic connections between adjacent dots"""
        for i in range(len(dots) - 1):
            self.draw_curved_line(ax, dots[i], dots[i + 1])
    
    def draw_curved_line(self, ax, start, end, control=None):
        """Draw a curved line between two points"""
        if control is None:
            # Simple curve with automatic control point
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            offset = 15
            control = (mid_x + offset, mid_y + offset)
        
        # Create Bezier curve points
        t_values = np.linspace(0, 1, 50)
        curve_x = []
        curve_y = []
        
        for t in t_values:
            x = (1-t)**2 * start[0] + 2*(1-t)*t * control[0] + t**2 * end[0]
            y = (1-t)**2 * start[1] + 2*(1-t)*t * control[1] + t**2 * end[1]
            curve_x.append(x)
            curve_y.append(y)
        
        ax.plot(curve_x, curve_y, 'b-', linewidth=2)

# Usage example and main analysis function
def analyze_kolam_from_image(image_path, save_analysis=True, recreate=True):
    """Complete analysis and recreation pipeline"""
    print("🎨 Starting Kolam Analysis...")
    
    # Initialize analyzer
    analyzer = KolamAnalyzer()
    
    try:
        # Load image
        print("📸 Loading image...")
        analyzer.load_image(image_path)
        
        # Detect dots
        print("🔍 Detecting dots...")
        dots = analyzer.detect_dots()
        print(f"   Found {len(dots)} dots")
        
        # Detect contours
        print("🔄 Detecting contours...")
        contours = analyzer.detect_contours()
        print(f"   Found {len(contours)} contours")
        
        # Analyze structure
        print("📐 Analyzing grid structure...")
        grid_size = analyzer.determine_grid_structure()
        print(f"   Grid size: {grid_size[0]} × {grid_size[1]}")
        
        # Check symmetry
        print("🔄 Checking symmetry...")
        is_symmetric = analyzer.check_symmetry()
        print(f"   Symmetric: {'Yes' if is_symmetric else 'No'}")
        
        # Extract design principles
        print("🧠 Extracting design principles...")
        principles = analyzer.extract_design_principles()
        
        # Visualize analysis
        if save_analysis:
            print("📊 Creating analysis visualization...")
            analyzer.visualize_analysis('kolam_analysis.png')
        
        # Save principles to JSON
        with open('kolam_principles.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_principles = principles.copy()
            for connection in json_principles['connections']:
                if 'contour' in connection:
                    connection.pop('contour', None)  # Remove non-serializable contour data
            json.dump(json_principles, f, indent=2, default=str)
        
        # Recreate Kolam
        if recreate:
            print("🎭 Recreating Kolam...")
            recreator = KolamRecreator(principles)
            recreator.recreate_kolam('recreated_kolam.png')
        
        print("✅ Analysis complete!")
        return principles
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "image5.png"
    
    # Run complete analysis
    principles = analyze_kolam_from_image(image_path)
    
    if principles:
        print("\n📋 Design Principles Summary:")
        print(f"   • Dots: {principles['dot_count']}")
        print(f"   • Contours: {principles['contour_count']}")
        print(f"   • Grid: {principles['grid_size'][0]} × {principles['grid_size'][1]}")
        print(f"   • Symmetric: {principles['is_symmetric']}")
        print(f"   • Pattern Type: {principles['pattern_type']}")
        
        print(f"\n📍 Dot Positions:")
        for i, pos in enumerate(principles['dot_positions']):
            print(f"   • Dot {i+1}: {pos}")