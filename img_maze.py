import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import random

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
    """Generate a maze from an input image."""
    global cols, rows, grid, w, x_offset, y_offset, walls, start_cell, end_cell

    if not file_path:
        messagebox.showerror("Error", "Please select an image first!")
        return

    # Load and process image
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    rows, cols = binary.shape
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
            is_wall = binary[j, i] == 0
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


def find_start_end_points():
    """Find valid start and end points from open boundary cells."""
    global start_cell, end_cell

    open_boundary_cells = []

    # Check the top and bottom rows
    for i in range(cols):
        if not grid[index(i, 0)].is_wall:
            open_boundary_cells.append(grid[index(i, 0)])
        if not grid[index(i, rows - 1)].is_wall:
            open_boundary_cells.append(grid[index(i, rows - 1)])

    # Check the left and right columns
    for j in range(rows):
        if not grid[index(0, j)].is_wall:
            open_boundary_cells.append(grid[index(0, j)])
        if not grid[index(cols - 1, j)].is_wall:
            open_boundary_cells.append(grid[index(cols - 1, j)])

    if len(open_boundary_cells) < 2:
        print("Error: No valid start and end points found!")
        return

    # Select start and end randomly ensuring they are different
    start_cell = random.choice(open_boundary_cells)
    open_boundary_cells.remove(start_cell)
    end_cell = random.choice(open_boundary_cells)


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
        canvas.create_oval(x + w // 4, y + w // 4, x + 3 * w // 4, y + 3 * w // 4, fill="green")

    if end_cell:
        x = end_cell.i * w + x_offset
        y = end_cell.j * w + y_offset
        canvas.create_oval(x + w // 4, y + w // 4, x + 3 * w // 4, y + 3 * w // 4, fill="red")



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
