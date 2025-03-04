import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import random
from collections import deque

# Global variables
cols, rows, grid, walls = 0, 0, [], []
w = 0
x_offset = 0
y_offset = 0
width, height = 600, 600  # Canvas size
file_path = None
start_cell = None
end_cell = None


def select_image():
    """Open file dialog to select an image."""
    global file_path
    file_path = filedialog.askopenfilename(title="Select a Maze Image",
                                           filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        print(f"Selected Image: {file_path}")
        generate_button.config(state=tk.NORMAL)  # Enable "Generate Maze" button
    else:
        print("No image selected.")
        generate_button.config(state=tk.DISABLED)  # Disable button if no image is selected


def index(i, j):
    """Return index of the cell in the grid."""
    if i < 0 or j < 0 or i >= cols or j >= rows:
        return None
    return i + j * cols


def generate_maze():
    """Generate a maze from an input image, removing outer padding."""
    global cols, rows, grid, w, x_offset, y_offset, walls, start_cell, end_cell

    if not file_path:
        messagebox.showerror("Error", "Please select an image first!")
        return

    # Load and process image
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Find the first and last row/column that contains a black pixel (wall)
    coords = np.column_stack(np.where(binary == 0))  # Get all black pixel coordinates
    if coords.size == 0:
        messagebox.showerror("Error", "No maze found in the image!")
        return

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop the image to remove white padding
    cropped_binary = binary[y_min:y_max + 1, x_min:x_max + 1]

    rows, cols = cropped_binary.shape
    w = min(width // cols, height // rows)

    x_offset = (width - cols * w) // 2
    y_offset = (height - rows * w) // 2

    grid.clear()
    walls.clear()

    class Cell:
        def __init__(self, i, j, is_wall):
            self.i, self.j = i, j
            self.is_wall = is_wall
            self.walls = [True, True, True, True]
            self.parent = self
            self.rank = 0

    for j in range(rows):
        for i in range(cols):
            is_wall = cropped_binary[j, i] == 0
            grid.append(Cell(i, j, is_wall))

    # Generate walls between cells
    for j in range(rows):
        for i in range(cols):
            cell = grid[index(i, j)]
            if not cell.is_wall:
                if i < cols - 1 and not grid[index(i + 1, j)].is_wall:
                    walls.append((cell, grid[index(i + 1, j)], 1))
                if j < rows - 1 and not grid[index(i, j + 1)].is_wall:
                    walls.append((cell, grid[index(i, j + 1)], 2))

    random.shuffle(walls)
    find_start_end_points()
    draw_maze()
    generate_button.config(state=tk.DISABLED)


def find_border_openings():
    """Detect all open cells on the maze border."""
    openings = []

    # Check top and bottom borders
    for i in range(cols):
        if not grid[index(i, 0)].is_wall:
            openings.append((i, 0))  # Top border
        if not grid[index(i, rows - 1)].is_wall:
            openings.append((i, rows - 1))  # Bottom border

    # Check left and right borders
    for j in range(rows):
        if not grid[index(0, j)].is_wall:
            openings.append((0, j))  # Left border
        if not grid[index(cols - 1, j)].is_wall:
            openings.append((cols - 1, j))  # Right border

    print(f"Detected Border Openings: {openings}")
    return openings


def bfs(start_cell):
    """Perform BFS to find the farthest point in the maze from the start cell."""
    visited = set()
    queue = deque([start_cell])
    visited.add(start_cell)

    # Direction vectors for moving (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    farthest_cell = start_cell

    while queue:
        current_cell = queue.popleft()

        # Check all 4 neighbors
        for dx, dy in directions:
            ni, nj = current_cell.i + dx, current_cell.j + dy

            if 0 <= ni < cols and 0 <= nj < rows and (ni, nj) not in visited:
                neighbor_cell = grid[index(ni, nj)]

                if not neighbor_cell.is_wall:
                    visited.add((ni, nj))
                    queue.append(neighbor_cell)

                    # Update farthest cell
                    farthest_cell = neighbor_cell

    return farthest_cell


def find_start_end_points():
    """Ensure start and end points are selected strictly from outer wall openings."""
    global start_cell, end_cell

    openings = find_border_openings()

    if len(openings) < 2:
        messagebox.showerror("Error", "Not enough border openings to set start and end points!")
        return

    # Randomly select a start position from the detected outer wall openings
    start_pos = random.choice(openings)
    start_cell = grid[index(start_pos[0], start_pos[1])]

    # Use BFS to find the farthest outer border opening from the start
    farthest_cell = bfs(start_cell)

    # Find the farthest valid border opening (ensuring it is an exit)
    valid_endings = [grid[index(x, y)] for x, y in openings if (x, y) != (start_cell.i, start_cell.j)]

    if valid_endings:
        # Choose the one farthest from the start
        end_cell = max(valid_endings, key=lambda cell: abs(cell.i - start_cell.i) + abs(cell.j - start_cell.j))
    else:
        end_cell = farthest_cell  # Fallback (should not happen if maze has at least 2 openings)

    print(f"Start: ({start_cell.i}, {start_cell.j}) (Green), End: ({end_cell.i}, {end_cell.j}) (Red)")


def detect_outer_walls():
    """Detect open cells on outer walls."""
    outer_walls = []

    for i in range(cols):
        if not grid[index(i, 0)].is_wall:
            outer_walls.append((i, 0, "Top"))
        if not grid[index(i, rows - 1)].is_wall:
            outer_walls.append((i, rows - 1, "Bottom"))

    for j in range(rows):
        if not grid[index(0, j)].is_wall:
            outer_walls.append((0, j, "Left"))
        if not grid[index(cols - 1, j)].is_wall:
            outer_walls.append((cols - 1, j, "Right"))

    return outer_walls


def draw_maze():
    """Draw the maze on the Tkinter canvas."""
    canvas.delete("all")

    for cell in grid:
        x = cell.i * w + x_offset
        y = cell.j * w + y_offset

        color = "black" if cell.is_wall else "white"
        canvas.create_rectangle(x, y, x + w, y + w, fill=color, outline=color)

    if start_cell:
        x = start_cell.i * w + x_offset
        y = start_cell.j * w + y_offset
        canvas.create_oval(x + 5, y + 5, x + w - 5, y + w - 5, fill="green")

    if end_cell:
        x = end_cell.i * w + x_offset
        y = end_cell.j * w + y_offset
        canvas.create_oval(x + 5, y + 5, x + w - 5, y + w - 5, fill="red")


# Create Tkinter window
root = tk.Tk()
root.title("Maze Generator")

top_frame = tk.Frame(root)
top_frame.pack()

image_button = tk.Button(top_frame, text="Select Image", command=select_image)
image_button.pack(side=tk.LEFT)

generate_button = tk.Button(top_frame, text="Generate Maze", command=generate_maze, state=tk.DISABLED)
generate_button.pack(side=tk.LEFT)

canvas = tk.Canvas(root, width=width, height=height)
canvas.pack()

root.mainloop()
