import heapq
import time
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import random
from collections import deque

from dfs_vision import solve_maze_dfs
from astar_vision import solve_maze_astar
from bfs_vision import solve_maze_bfs

# Global variables
cols, rows, grid, walls = 0, 0, [], []
w = 0
x_offset = 0
y_offset = 0
width, height = 600, 600  # Canvas size
file_path = None
start = None
goal = None

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
    global cols, rows, grid, w, x_offset, y_offset, walls, start, goal
    start_time = time.time()
    if not file_path:
        messagebox.showerror("Error", "Please select an image first!")
        return

    # Load and process image
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Find the first and last row/column that contains a black pixel (walls)
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
    print(w)
    print(rows)
    print(cols)

    
    x_offset = (width - cols * w) // 2
    y_offset = (height - rows * w) // 2

    grid.clear()
    walls.clear()

    vertical_opening = find_opening_width(cropped_binary, axis=0)  # Check left/right openings
    horizontal_opening = find_opening_width(cropped_binary, axis=1)  # Check top/bottom openings

    print(f"Vertical Opening Width: {vertical_opening} pixels")
    print(f"Horizontal Opening Width: {horizontal_opening} pixels")

    class Cell:
        def __init__(self, i, j, is_wall):
            self.i, self.j = i, j
            self.is_wall = is_wall
            self.walls = [True, True, True, True]
            self.parent = self
            self.rank = 0
        
        def __lt__(self, other):
            # Compare cells based on their coordinates (or any unique identifier)
             return (self.i, self.j) < (other.i, other.j)
    
    for j in range(rows):
        for i in range(cols):
            is_wall = cropped_binary[j, i] == 0
            grid.append(Cell(i, j, is_wall))
            i+=min(vertical_opening,horizontal_opening)
        j+=min(vertical_opening,horizontal_opening)

    # Generate walls between cells
    for j in range(rows):
        for i in range(cols):
            cell = grid[index(i, j)]
            if not cell.is_wall:
                if i < cols - 1 and not grid[index(i + 1, j)].is_wall:
                    walls.append((cell, grid[index(i + 1, j)], 1))
                if j < rows - 1 and not grid[index(i, j + 1)].is_wall:
                    walls.append((cell, grid[index(i, j + 1)], 2))
            i+=min(vertical_opening,horizontal_opening)
        j+=min(vertical_opening,horizontal_opening)

    random.shuffle(walls)
    find_start_end_points()
    draw_maze()
    generate_button.config(state=tk.DISABLED)
    execution_time_label.config(text=f"Execution Time: {time.time() - start_time:.4f}s")

# Find the number of pixels in the entrance and exit openings
def find_opening_width(binary, axis=0):
    """Finds the width of an opening along a given axis (0 = vertical, 1 = horizontal)."""
    if axis == 0:  # Vertical opening (left or right side of the maze)
        for col in range(binary.shape[1]):  # Scan columns from left
            column = binary[:, col]
            white_pixels = np.sum(column == 255)
            if white_pixels > 0:  # Found an opening
                return white_pixels
    elif axis == 1:  # Horizontal opening (top or bottom)
        for row in range(binary.shape[0]):  # Scan rows from top
            row_pixels = binary[row, :]
            white_pixels = np.sum(row_pixels == 255)
            if white_pixels > 0:
                return white_pixels
    return 0  # No opening found


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
    """Ensure start and end points take the width of one full cell."""
    global start, goal

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

    start = grid[index(*start_opening[len(start_opening) // 2])] if start_opening else None
    goal = grid[index(*end_opening[len(end_opening) // 2])] if end_opening else None

    if not start or not goal:
        messagebox.showerror("Error", "Could not find valid start or end points!")


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

    # Increase the size of start and end cells
    extra_size = w * 15  # Increase size by 30% of the cell width

    if start:
        x = start.i * w + x_offset
        y = start.j * w + y_offset
        canvas.create_rectangle(
            x - extra_size / 2, y - extra_size / 2,
            x + w + extra_size / 2, y + w + extra_size / 2,
            fill="green", outline="green"
        )

    if goal:
        x = goal.i * w + x_offset
        y = goal.j * w + y_offset
        canvas.create_rectangle(
            x - extra_size / 2, y - extra_size / 2,
            x + w + extra_size / 2, y + w + extra_size / 2,
            fill="red", outline="red"
        )
        
    canvas.update()


class Cell:
    def __init__(self, i, j, is_wall):
        self.i = i  # Column index of the cell
        self.j = j  # Row index of the cell
        self.is_wall = is_wall  # Whether the cell is a wall (True/False)
        self.walls = [True, True, True, True]  # Walls (top, right, bottom, left)
        self.parent = self  # Parent cell for union-find operations
        self.rank = 0  # Rank for union-find operations

    def __lt__(self, other):
        # Compare cells based on their coordinates (or any unique identifier)
        return (self.i, self.j) < (other.i, other.j)
    

#-------------------------------- using 3x3 neighborhood -----------------------------------------
def draw_path(path):
    """Draw the final path on the canvas, slightly moved away from edges."""
    if not path:
        return  # No path to draw

    adjusted_path = []
    for cell in path:
        total_i, total_j, count = 0, 0, 0

        # Check a 3x3 neighborhood to find open space
        for di in range(-1, 2):  # -1, 0, 1
            for dj in range(-1, 2):  # -1, 0, 1
                ni, nj = cell.i + di, cell.j + dj
                if 0 <= ni < cols and 0 <= nj < rows:
                    neighbor_cell = grid[index(ni, nj)]
                    if not neighbor_cell.is_wall:
                        total_i += ni
                        total_j += nj
                        count += 1

        # Calculate a smooth shift towards the center
        if count > 0:
            center_i = (total_i / count) * 0.7 + cell.i * 0.3
            center_j = (total_j / count) * 0.7 + cell.j * 0.3
        else:
            center_i, center_j = cell.i, cell.j  # Keep original if no open space

        adjusted_cell = Cell(int(center_i), int(center_j), cell.is_wall)
        adjusted_path.append(adjusted_cell)

    def draw_step(i):
        """Draw the path step by step in real-time with a smooth offset."""
        if i < len(adjusted_path) - 1:
            x1 = adjusted_path[i].i * w + x_offset + w * 0.6  # Slightly offset
            y1 = adjusted_path[i].j * w + y_offset + w * 0.6
            x2 = adjusted_path[i + 1].i * w + x_offset + w * 0.6
            y2 = adjusted_path[i + 1].j * w + y_offset + w * 0.6

            canvas.create_line(x1, y1, x2, y2, fill="blue", width=1)
            root.update()
            root.after(1, draw_step, i + 1)  # Adjust speed if needed

    draw_step(0)  # Start animation


def solve_maze_selected():
    # Clear previous path
    draw_maze()
    if algorithm_var.get() == "BFS":
        solve_maze_bfs(start,goal, draw_path,execution_time_label,grid,rows,cols,index)
    elif algorithm_var.get() == "A*":
        solve_maze_astar(start,goal, draw_path,execution_time_label,grid,rows,cols,index)
    else:
        solve_maze_dfs(start,goal, draw_path,execution_time_label,grid,rows,cols,index)


# Create Tkinter window
root = tk.Tk()
root.title("Maze Generator")

top_frame = tk.Frame(root)
top_frame.pack()

image_button = tk.Button(top_frame, text="Select Image", command=select_image)
image_button.pack(side=tk.LEFT)

generate_button = tk.Button(top_frame, text="Generate Maze", command=generate_maze, state=tk.DISABLED)
generate_button.pack(side=tk.LEFT)

# create time label
execution_time_label = tk.Label(root, text="Execution Time: 0.0s", font=('Arial', 12))
execution_time_label.pack(pady=10)

# create maze solve button
solve_button = tk.Button(top_frame, text="Solve Maze", command=solve_maze_selected)
solve_button.pack(side=tk.RIGHT, padx=10)

# Algorithm selection radio buttons
algo_frame = tk.Frame(top_frame, bg="#d3d3d3")
algo_frame.pack(side=tk.RIGHT)

# Algorithm Selection (BFS, DFS, A*)
algorithm_var = tk.StringVar(value="BFS")  # Default selection is BFS

astar_radio = tk.Radiobutton(algo_frame, text="A*", variable=algorithm_var, value="A*", bg="#d3d3d3")
astar_radio.pack(side=tk.RIGHT, padx=5)

dfs_radio = tk.Radiobutton(algo_frame, text="DFS", variable=algorithm_var, value="DFS", bg="#d3d3d3")
dfs_radio.pack(side=tk.RIGHT, padx=5)

bfs_radio = tk.Radiobutton(algo_frame, text="BFS", variable=algorithm_var, value="BFS", bg="#d3d3d3")
bfs_radio.pack(side=tk.RIGHT, padx=5)

# Algorithm selection
algo_label = tk.Label(top_frame, text="Algorithm:", bg="#d3d3d3")
algo_label.pack(side=tk.RIGHT, padx=5)

# draw canvas to create maze
canvas = tk.Canvas(root, width=width, height=height)
canvas.pack()

root.mainloop()