import cv2
import numpy as np
import pygame

# Load the image
image = cv2.imread('maze.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to smooth edges
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Convert to binary (white = path, black = walls)
_, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

# Use morphological operations to clean noise
kernel = np.ones((3,3), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Find contours of the maze
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assume the largest contour is the maze
maze_contour = max(contours, key=cv2.contourArea)

# Create a mask for the maze
mask = np.zeros_like(gray)
cv2.drawContours(mask, [maze_contour], -1, 255, thickness=cv2.FILLED)

# Get the bounding rectangle of the maze
x, y, w, h = cv2.boundingRect(maze_contour)

# Identify open border cells only on the edges (walls)
open_cells = []

def is_valid_opening(px, py):
    """ Ensure pixel is on the border and part of the path """
    if binary[py, px] == 255 and mask[py, px] == 255:
        # Check if it's actually on the outermost boundary
        return px == x or px == x + w - 1 or py == y or py == y + h - 1
    return False


# Check top and bottom borders
for i in range(x, x + w):
    if is_valid_opening(i, y):
        open_cells.append((i, y))  # Top border
    if is_valid_opening(i, y + h - 1):
        open_cells.append((i, y + h - 1))  # Bottom border

# Check left and right borders
for j in range(y, y + h):
    if is_valid_opening(x, j):
        open_cells.append((x, j))  # Left border
    if is_valid_opening(x + w - 1, j):
        open_cells.append((x + w - 1, j))  # Right border

# Select start and goal only from valid wall openings
if len(open_cells) >= 2:
    start, goal = open_cells[0], open_cells[-1]
else:
    start, goal = None, None

print(f"Start Cell: {start}, Goal Cell: {goal}")

# Draw start and goal on the image
if start:
    cv2.circle(image, start, 5, (0, 255, 0), -1)  # Green for start
if goal:
    cv2.circle(image, goal, 5, (0, 0, 255), -1)  # Red for goal

# Save the image with start and goal marked
cv2.imwrite('maze_with_start_goal.png', image)

# Initialize Pygame
pygame.init()
window_size = (image.shape[1], image.shape[0])
screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
pygame.display.set_caption("Maze with Start and Goal")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pygame_image = pygame.surfarray.make_surface(image_rgb.swapaxes(0, 1))

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.VIDEORESIZE:
            new_width, new_height = event.w, event.h
            screen = pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)

    screen.blit(pygame.transform.scale(pygame_image, window_size), (0, 0))
    pygame.display.flip()

pygame.quit()
