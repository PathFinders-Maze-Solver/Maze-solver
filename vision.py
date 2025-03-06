import heapq
import time
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

def generate_maze():
    """Generate a maze from an input image, removing outer padding."""
    global cols, rows, grid, w, x_offset, y_offset, walls, start, goal

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


def get_neighbors(cell):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors = []
    for dx, dy in directions:
        ni, nj = cell.i + dx, cell.j + dy
        if 0 <= ni < cols and 0 <= nj < rows:
            neighbor = grid[index(ni, nj)]
            if not neighbor.is_wall:
                neighbors.append(neighbor)
    return neighbors

def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(current)  # Add the start point
    path.reverse()  # Reverse to get the path from start to goal
    return path


def draw_solution(path, solving):
    """
    Draw the solution path one step at a time to show real-time steps.

    :param path: List of cells in the solution path.
    :param solving: If True, draw the solution path in blue. If False, reset the cells to white.
    """
    def draw_step(i):
        if i < len(path):
            cell = path[i]
            x = cell.i * w + x_offset
            y = cell.j * w + y_offset
            color = "blue" if solving else "white"  # Blue for solving, white for resetting
            canvas.create_rectangle(x, y, x + w, y + w, fill=color, outline=color)
            root.update()  # Update the canvas to show the current step

            # Schedule the next step
            root.after(1, draw_step, i + 1)  # Adjust 100ms for desired speed

    draw_step(0)  # Start drawing from the first step of the path


def solve_maze_bfs():
    start_time = time.time()
    queue = deque([start])
    came_from = {}
    visited = set()
    visited.add(start)
    
    while queue:
        current = queue.popleft()
        if current == goal:
            path = reconstruct_path(came_from, goal)
            draw_solution(path, solving=True)
            execution_time_label.config(text=f"Execution Time: {time.time() - start_time:.4f}s")
            return
        
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)


def solve_maze_dfs():
    """Solve the maze using DFS and draw the solution path after completion."""
    start_time = time.time()
    stack = [(start, [start])]  # Stack stores tuples of (current cell, current path)
    visited = set()  # Set to keep track of visited cells
    visited.add(start)

    while stack:
        current, path = stack.pop()  # Get the last cell and its path from the stack
        if current == goal:
            # Once the goal is reached, draw the entire solution path
            draw_solution(path, solving=True)
            execution_time_label.config(text=f"Execution Time: {time.time() - start_time:.4f}s")
            return  # Stop once the goal is found

        # Explore all neighbors
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append((neighbor, path + [neighbor]))  # Add the neighbor and the updated path to the stack

    # If no path is found, indicate failure (optional)
    print("No path found.")
    execution_time_label.config(text="Execution Time: No path found")

                
def solve_maze_astar():
    """Solve the maze using A* algorithm."""
    start_time = time.time()
    open_set = [(0, start)]  # Priority queue of (f_score, cell)
    came_from = {}
    g_score = {cell: float('inf') for cell in grid}
    g_score[start] = 0
    f_score = {cell: float('inf') for cell in grid}
    f_score[start] = abs(start.i - goal.i) + abs(start.j - goal.j)

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = reconstruct_path(came_from, goal)
            draw_solution(path, solving=True)
            execution_time_label.config(text=f"Execution Time: {time.time() - start_time:.4f}s")
            return

        for neighbor in get_neighbors(current):
            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + abs(neighbor.i - goal.i) + abs(neighbor.j - goal.j)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
                
def solve_maze_selected():
    # Clear previous path
    draw_maze()
    if algorithm_var.get() == "BFS":
        solve_maze_bfs()
    elif algorithm_var.get() == "A*":
        solve_maze_astar()
    else:
        solve_maze_dfs()

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