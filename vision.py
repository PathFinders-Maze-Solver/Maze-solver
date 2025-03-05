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
    """Ensure start and end points are selected strictly from outer wall openings and placed in the middle of the opening area."""
    global start, goal

    openings = find_border_openings()

    if len(openings) < 2:
        messagebox.showerror("Error", "Not enough border openings to set start and end points!")
        return

    # Function to find the middle cell of an opening area
    def find_middle_cell(opening_cells):
        if not opening_cells:
            return None
        # Group cells by row or column based on the border
        if opening_cells[0][1] == 0 or opening_cells[0][1] == rows - 1:  # Top or bottom border
            # Group by column and find the middle row
            col = opening_cells[0][0]
            rows_in_opening = [cell[1] for cell in opening_cells if cell[0] == col]
            middle_row = rows_in_opening[len(rows_in_opening) // 2]
            return (col, middle_row)
        else:  # Left or right border
            # Group by row and find the middle column
            row = opening_cells[0][1]
            cols_in_opening = [cell[0] for cell in opening_cells if cell[1] == row]
            middle_col = cols_in_opening[len(cols_in_opening) // 2]
            return (middle_col, row)

    # Randomly select a start position from the detected outer wall openings
    start_pos = random.choice(openings)
    start_cells = [cell for cell in openings if cell[0] == start_pos[0] or cell[1] == start_pos[1]]
    start_middle = find_middle_cell(start_cells)
    if start_middle:
        start = grid[index(start_middle[0], start_middle[1])]

    # Use BFS to find the farthest outer border opening from the start
    farthest_cell = bfs(start)

    # Find the farthest valid border opening (ensuring it is an exit)
    valid_endings = [grid[index(x, y)] for x, y in openings if (x, y) != (start.i, start.j)]
    if valid_endings:
        # Choose the one farthest from the start
        goal_cell = max(valid_endings, key=lambda cell: abs(cell.i - start.i) + abs(cell.j - start.j))
        goal_cells = [cell for cell in openings if cell[0] == goal_cell.i or cell[1] == goal_cell.j]
        goal_middle = find_middle_cell(goal_cells)
        if goal_middle:
            goal = grid[index(goal_middle[0], goal_middle[1])]
    else:
        goal = farthest_cell  # Fallback (should not happen if maze has at least 2 openings)

    print(f"Start: ({start.i}, {start.j}) (Green), End: ({goal.i}, {goal.j}) (Red)")

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

    if start:
        x = start.i * w + x_offset
        y = start.j * w + y_offset
        canvas.create_oval(x + 5, y + 5, x + w - 5, y + w - 5, fill="green")

    if goal:
        x = goal.i * w + x_offset
        y = goal.j * w + y_offset
        canvas.create_oval(x + 5, y + 5, x + w - 5, y + w - 5, fill="red")

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
    path.reverse()
    return path

def draw_path(path):
    for i in range(len(path) - 1):
        x1, y1 = path[i].i * w + x_offset + w // 2, path[i].j * w + y_offset + w // 2
        x2, y2 = path[i + 1].i * w + x_offset + w // 2, path[i + 1].j * w + y_offset + w // 2
        canvas.create_line(x1, y1, x2, y2, fill="blue", width=2)

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
            draw_path(path)
            execution_time_label.config(text=f"Execution Time: {time.time() - start_time:.4f}s")
            return
        
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

def solve_maze_dfs():
    start_time = time.time()
    stack = [start]  # Stack stores cells to explore
    came_from = {}  # Dictionary to store the parent of each cell
    visited = set()  # Set to keep track of visited cells
    visited.add(start)

    while stack:
        current = stack.pop()  # Get the last cell from the stack
        if current == goal:
            # Reconstruct and draw the path
            path = reconstruct_path(came_from, goal)
            draw_path(path)
            execution_time_label.config(text=f"Execution Time: {time.time() - start_time:.4f}s")
            return  # Stop once the goal is found

        # Explore all neighbors
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current  # Record the parent of the neighbor
                stack.append(neighbor)  # Add the neighbor to the stack
                
def solve_maze_astar():
    start_time = time.time()
    open_set = [(0, start)]
    came_from = {}
    g_score = {cell: float('inf') for cell in grid}
    g_score[start] = 0
    f_score = {cell: float('inf') for cell in grid}
    f_score[start] = abs(start.i - goal.i) + abs(start.j - goal.j)
    
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = reconstruct_path(came_from, goal)
            draw_path(path)
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