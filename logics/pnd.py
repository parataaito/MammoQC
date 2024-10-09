import numpy as np
import cv2
from scipy import stats

def pectoral_nipple_distance(img_path, mask_path, xtl, ytl, xbr, ybr, orientation, view):
    # Load the image and mask
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if view != "CC" else None
    
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
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_color, 'N', (x_nipple+10, y_nipple), font, 0.5, (255, 0, 0), 2)
    cv2.putText(img_color, 'P', (x_intersect+10, y_intersect), font, 0.5, (0, 255, 255), 2)
    cv2.putText(img_color, f'Mammogram Analysis - {orientation} {view}', (10, 30), font, 1, (255, 255, 255), 2)
    
    return img_color, (x_intersect, y_intersect), distance

if __name__ == "__main__":
    # List of orientations and views
    orientations = ["Right", "Left"]
    views = ["CC", "MLO"]
    
    for orientation in orientations:
        for view in views:
            if orientation == "Right" and view == "MLO":
                img_path = r'D:\Code\dcm2png\png_data\00efff24259203a7f19a2875295e4ba6_04d59c956abf0bc15c0fd6dec47374b5.png'
                mask_path = r'D:\Code\dcm2png\png_masks\00efff24259203a7f19a2875295e4ba6_04d59c956abf0bc15c0fd6dec47374b5.png'
                xtl, ytl, xbr, ybr = 1562.39, 2345.40, 1756.72, 2559.83
            elif orientation == "Left" and view == "MLO":
                img_path = r'D:\Code\dcm2png\png_data\e8a024b2a99a8b12b25a1e436556d0d2_04afe127c8d6daf4d27d227c64fa8aea.png'
                mask_path = r'D:\Code\dcm2png\png_masks\e8a024b2a99a8b12b25a1e436556d0d2_04afe127c8d6daf4d27d227c64fa8aea.png'
                xtl, ytl, xbr, ybr = 839.46, 1840.58, 1008.32, 2057.69
            elif orientation == "Left" and view == "CC":
                img_path = r'D:\Code\dcm2png\png_data\033e4f9ee05749cd591c958aa873dc8b_00c3c05f7ff415d71fae16ae999c178d.png'
                mask_path = r'D:\Code\dcm2png\png_masks\033e4f9ee05749cd591c958aa873dc8b_00c3c05f7ff415d71fae16ae999c178d.png'
                xtl, ytl, xbr, ybr = 677.86, 1390.51, 805.18, 1544.63
            else: # orientation == "Right" and view == "CC":
                img_path = r'D:\Code\dcm2png\png_data\0b0241e1676978a4e5717cb6406f48d1_00dfcde5aaf6cd0aab3c3a0435632b3f.png'
                mask_path = r'D:\Code\dcm2png\png_masks\0b0241e1676978a4e5717cb6406f48d1_00dfcde5aaf6cd0aab3c3a0435632b3f.png'
                xtl, ytl, xbr, ybr = 1729.91, 1393.86, 1837.13, 1568.09

            img_color, intersection_point, distance = pectoral_nipple_distance(img_path, mask_path, xtl, ytl, xbr, ybr, orientation, view)
            print(f"Intersection point P: {intersection_point}")
            print(f"Distance between N and P: {distance}")
            
            # Save the image
            output_path = f'pnd_{orientation}_{view}.png'
            cv2.imwrite(output_path, img_color)
            
            print(f"Analysis visualization saved as: {output_path}")