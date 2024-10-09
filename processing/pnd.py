import numpy as np
import cv2
from scipy import stats
import argparse

def pectoral_nipple_distance(img, mask, xtl, ytl, xbr, ybr, orientation, view):
    assert view in ["CC", "MLO"], "Invalid view. Must be 'CC' or 'MLO'."
    assert orientation in ["Left", "Right"], "Invalid orientation. Must be 'Left' or 'Right'."
    
    # Create a color version of the image for drawing
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    rows, cols = img.shape
    
    if view == "CC":  # For CC view, don't use the mask
        # Use the first or last column of the image as L1 based on orientation
        if orientation == "Left":
            x = np.zeros(rows, dtype=int)
        else:  # "Right"
            x = np.full(rows, cols - 1, dtype=int)
        y = np.arange(rows)
        slope = float('inf')  # Vertical line
        intercept = x[0]  # First x value
    else:  # For MLO view
        # Keep only the leftmost or rightmost white pixel for each row based on orientation
        edge_pixels = []
        for row in range(rows):
            white_pixels = np.where(mask[row] == 255)[0]
            if len(white_pixels) > 0:
                if orientation == "Right":
                    edge_pixels.append((row, white_pixels[0]))  # Leftmost pixel
                elif orientation == "Left":
                    edge_pixels.append((row, white_pixels[-1]))  # Rightmost pixel
        
        # Fit a line to the edge pixels (L1)
        y, x = zip(*edge_pixels)
        slope, intercept, _, _, _ = stats.linregress(x, y)
    
    # Function to get a point on L1
    def point_on_L1(x):
        if slope == float('inf'):
            return int(intercept)  # Return the x-coordinate for vertical line
        return int(slope * x + intercept)
    
    # Calculate center of bounding box (N)
    x_nipple = int((xtl + xbr) / 2)
    y_nipple = int((ytl + ybr) / 2)
    
    # Calculate perpendicular slope for L2
    if slope == float('inf'):
        perp_slope = 0
        perp_intercept = y_nipple
    elif slope == 0:
        perp_slope = float('inf')
        perp_intercept = x_nipple
    else:
        perp_slope = -1 / slope
        perp_intercept = y_nipple - perp_slope * x_nipple
        
    # Find intersection point P
    if slope == float('inf'):
        x_intersect = int(intercept)
        y_intersect = y_nipple
    elif perp_slope == float('inf'):
        x_intersect = x_nipple
        y_intersect = int(slope * x_nipple + intercept)
    else:
        x_intersect = int((perp_intercept - intercept) / (slope - perp_slope))
        y_intersect = int(slope * x_intersect + intercept)
    
    # Calculate distance between N and P
    distance = np.sqrt((x_intersect - x_nipple)**2 + (y_intersect - y_nipple)**2)
    
    # Draw on the image
    # Draw L1
    if slope == float('inf'):
        cv2.line(img_color, (int(intercept), 0), (int(intercept), rows-1), (0, 0, 255), 2)
    else:
        y1 = int(slope * 0 + intercept)
        y2 = int(slope * (cols-1) + intercept)
        cv2.line(img_color, (0, y1), (cols-1, y2), (0, 0, 255), 2)
   
    # Draw L2
    if perp_slope == float('inf'):
        cv2.line(img_color, (x_nipple, 0), (x_nipple, rows-1), (0, 255, 0), 2)
    else:
        y1 = int(perp_slope * 0 + perp_intercept)
        y2 = int(perp_slope * (cols-1) + perp_intercept)
        cv2.line(img_color, (0, y1), (cols-1, y2), (0, 255, 0), 2)
    
    # Draw points
    cv2.circle(img_color, (x_nipple, y_nipple), 5, (255, 0, 0), -1)  # N (Nipple)
    cv2.circle(img_color, (x_intersect, y_intersect), 5, (0, 255, 255), -1)  # P (Intersection)
    
    # Draw bounding box
    cv2.rectangle(img_color, (int(xtl), int(ytl)), (int(xbr), int(ybr)), (255, 0, 255), 2)
    
    # Draw edge pixels (only for MLO view)
    if view != "CC":
        for pixel in zip(y, x):
            cv2.circle(img_color, (pixel[1], pixel[0]), 1, (255, 255, 0), -1)
    
    return img_color, (x_intersect, y_intersect), distance

def main(args):
    # Load the image and mask
    img = cv2.imread(args.img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE) if args.view != "CC" else None
    
    img_color, intersection_point, distance = pectoral_nipple_distance(
        img, mask, args.xtl, args.ytl, args.xbr, args.ybr, 
        args.orientation, args.view
    )
    print(f"Intersection point P: {intersection_point}")
    print(f"Distance between N and P: {distance}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform pectoral-nipple distance analysis on mammogram images")
    parser.add_argument("--img_path", type=str, required=True, help="Path to the mammogram image")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to the mask image (not used for CC view)")
    parser.add_argument("--xtl", type=float, required=True, help="X-coordinate of top-left corner of nipple bounding box")
    parser.add_argument("--ytl", type=float, required=True, help="Y-coordinate of top-left corner of nipple bounding box")
    parser.add_argument("--xbr", type=float, required=True, help="X-coordinate of bottom-right corner of nipple bounding box")
    parser.add_argument("--ybr", type=float, required=True, help="Y-coordinate of bottom-right corner of nipple bounding box")
    parser.add_argument("--orientation", type=str, choices=["Right", "Left"], required=True, help="Orientation of the mammogram")
    parser.add_argument("--view", type=str, choices=["CC", "MLO"], required=True, help="View of the mammogram")
    
    args = parser.parse_args()
    main(args)