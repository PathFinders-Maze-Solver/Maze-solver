import cv2
import numpy as np

# Load the image
image = cv2.imread('maze.png')

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color ranges for green and red in HSV
green_lower = np.array([35, 50, 50])
green_upper = np.array([85, 255, 255])

red_lower1 = np.array([0, 50, 50])
red_upper1 = np.array([10, 255, 255])

red_lower2 = np.array([170, 50, 50])
red_upper2 = np.array([180, 255, 255])

# Create masks for green and red colors
green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
red_mask1 = cv2.inRange(hsv_image, red_lower1, red_upper1)
red_mask2 = cv2.inRange(hsv_image, red_lower2, red_upper2)

# Combine red masks
red_mask = cv2.bitwise_or(red_mask1, red_mask2)

# Find contours for green and red arrows
green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Function to get the center of the arrow
def get_center(contour):
    moments = cv2.moments(contour)
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    return (cx, cy)

# Detect the start and end points based on color and position
start_point = None
end_point = None

for contour in green_contours:
    start_point = get_center(contour)
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

for contour in red_contours:
    end_point = get_center(contour)
    cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)

# Display the start and end points on the image
if start_point:
    cv2.circle(image, start_point, 10, (0, 255, 0), -1)  # Green circle for start
    print(f"Start Point: {start_point}")

if end_point:
    cv2.circle(image, end_point, 10, (0, 0, 255), -1)  # Red circle for end
    print(f"End Point: {end_point}")

# Show the processed image
cv2.imshow('Maze Image with Arrows', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
