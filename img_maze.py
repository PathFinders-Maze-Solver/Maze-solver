import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
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
        generate_button.config(state=tk.NORMAL)  # Enable "Generate Maze" button
    else:
        generate_button.config(state=tk.DISABLED)  # Disable button if no image is selected


def index(i, j):
    """Return index of the cell in the grid."""
    if i < 0 or j < 0 or i >= cols or j >= rows:
        return None
    return i + j * cols


def find_center_of_opening(openings):
    """Find the center-most open cell from a given list of open positions."""
    if not openings:
        return None
    return grid[index(*openings[len(openings) // 2])]


def find_start_end_points():
    """Ensure start and end points take the width of one full cell."""
    global start_cell, end_cell

    def find_valid_openings(edge_cells):
        """Find continuous openings at least one cell wide."""
        valid_openings = []
        temp_opening = []

        for i, j in edge_cells:
            if not grid[index(i, j)].is_wall:
                temp_opening.append((i, j))
            else:
                if len(temp_opening) >= w:  # Ensure it's at least 1 cell wide
                    valid_openings.append(temp_opening)
                temp_opening = []

        if len(temp_opening) >= w:
            valid_openings.append(temp_opening)

        return valid_openings

    # Find valid openings at least one cell wide
    top_openings = find_valid_openings([(i, 0) for i in range(cols)])
    bottom_openings = find_valid_openings([(i, rows - 1) for i in range(cols)])
    left_openings = find_valid_openings([(0, j) for j in range(rows)])
    right_openings = find_valid_openings([(cols - 1, j) for j in range(rows)])

    # Choose the central cell of the widest opening
    start_opening = top_openings[0] if top_openings else (left_openings[0] if left_openings else [])
    end_opening = bottom_openings[0] if bottom_openings else (right_openings[0] if right_openings else [])

    start_cell = grid[index(*start_opening[len(start_opening) // 2])] if start_opening else None
    end_cell = grid[index(*end_opening[len(end_opening) // 2])] if end_opening else None

    if not start_cell or not end_cell:
        messagebox.showerror("Error", "Could not find valid start or end points!")


def generate_maze():
    """Generate a maze from an input image, removing outer padding."""
    global cols, rows, grid, w, x_offset, y_offset, walls, start_cell, end_cell

    if not file_path:
        messagebox.showerror("Error", "Please select an image first!")
        return

    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    coords = np.column_stack(np.where(binary == 0))  # Get all black pixel coordinates
    if coords.size == 0:
        messagebox.showerror("Error", "No maze found in the image!")
        return

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    cropped_binary = binary[y_min:y_max + 1, x_min:x_max + 1]
    rows, cols = cropped_binary.shape

    w = min(20, width // cols, height // rows)  # Increase cell size, default to 20
    x_offset = (width - cols * w) // 2
    y_offset = (height - rows * w) // 2

    grid.clear()
    walls.clear()

    class Cell:
        def __init__(self, i, j, is_wall):
            self.i, self.j = i, j
            self.is_wall = is_wall

    for j in range(rows):
        for i in range(cols):
            is_wall = cropped_binary[j, i] == 0
            grid.append(Cell(i, j, is_wall))

    find_start_end_points()
    draw_maze()
    generate_button.config(state=tk.DISABLED)
    solve_button.config(state=tk.NORMAL)


def solve_maze_bfs():
    """Solve the maze using BFS and visualize the path."""
    if not start_cell or not end_cell:
        messagebox.showerror("Error", "Start or End cell not set!")
        return

    queue = deque([(start_cell, [])])
    visited = set()
    visited.add(start_cell)

    while queue:
        current_cell, path = queue.popleft()
        path.append(current_cell)

        if current_cell == end_cell:
            draw_solution(path)
            return

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            ni, nj = current_cell.i + dx, current_cell.j + dy
            neighbor_index = index(ni, nj)
            if neighbor_index is not None:
                neighbor_cell = grid[neighbor_index]
                if not neighbor_cell.is_wall and neighbor_cell not in visited:
                    visited.add(neighbor_cell)
                    queue.append((neighbor_cell, path.copy()))

    messagebox.showinfo("Maze Solver", "No path found!")


def draw_solution(path):
    """Draw the solution path on the maze."""
    for cell in path:
        x = cell.i * w + x_offset
        y = cell.j * w + y_offset
        canvas.create_rectangle(x, y, x + w, y + w, fill="blue", outline="blue")
    canvas.update()


def draw_maze():
    """Draw the maze on the Tkinter canvas."""
    canvas.delete("all")

    for cell in grid:
        x = cell.i * w + x_offset
        y = cell.j * w + y_offset
        color = "black" if cell.is_wall else "white"
        canvas.create_rectangle(x, y, x + w, y + w, fill=color, outline=color)

    # Increase the size of start and end cells
    extra_size = w * 0.3  # Increase size by 30% of the cell width

    if start_cell:
        x = start_cell.i * w + x_offset
        y = start_cell.j * w + y_offset
        canvas.create_rectangle(
            x - extra_size / 2, y - extra_size / 2,
            x + w + extra_size / 2, y + w + extra_size / 2,
            fill="green", outline="green"
        )

    if end_cell:
        x = end_cell.i * w + x_offset
        y = end_cell.j * w + y_offset
        canvas.create_rectangle(
            x - extra_size / 2, y - extra_size / 2,
            x + w + extra_size / 2, y + w + extra_size / 2,
            fill="red", outline="red"
        )

    canvas.update()



root = tk.Tk()
root.title("Maze Generator")

top_frame = tk.Frame(root)
top_frame.pack()

image_button = tk.Button(top_frame, text="Select Image", command=select_image)
image_button.pack(side=tk.LEFT)

generate_button = tk.Button(top_frame, text="Generate Maze", command=generate_maze, state=tk.DISABLED)
generate_button.pack(side=tk.LEFT)

solve_button = tk.Button(top_frame, text="Solve Maze", command=solve_maze_bfs, state=tk.DISABLED)
solve_button.pack(side=tk.LEFT)

canvas = tk.Canvas(root, width=width, height=height)
canvas.pack()

root.mainloop()
