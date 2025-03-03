# Maze-solver

## Process
1. Reads the maze image
2. Converts the image to grayscale to simplify further processing
3. A Gaussian blur is applied to smooth edges and reduce noise.
4. Converts the image into a binary (black and white) format
5. Morphological closing (cv2.MORPH_CLOSE) removes small noise and closes gaps in paths
6. Finds the outer boundaries (contours) of the maze.
cv2.RETR_EXTERNAL retrieves only the external contour, ignoring internal components.
cv2.CHAIN_APPROX_SIMPLE compresses contour points to save memory.
Assumes the largest contour represents the maze boundary.
7. Create a Mask for the Maze
8. Get the Bounding Box of the Maze
9. Identify Openings on the Borders (Entry and Exit)
Iterates through the image’s edges (top, bottom, left, and right).
Identifies valid open paths as potential start and goal points.
If at least two valid openings exist, the first is the start, and the last is the goal.
10. Draw Start and Goal Points on the Image
11. Display the Image Using Pygame
12. Convert Image to Pygame Format and Display the Image in a Pygame Window.

