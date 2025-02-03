from queue import PriorityQueue
import heapq
import time
from turtle import Screen
from networkx import neighbors
import pygame
import sys
import random
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from collections import deque

pygame.init()

# Global variables for maze state
cols, rows, grid, stack, current, start, goal = 0, 0, [], [], None, None, None
w = 0
x_offset = 0
y_offset = 0
width = 600  # Adjust to a smaller value for your pygame window
height = 600  # Adjust accordingly


def generate_maze():
    global cols, rows, grid, stack, current, start, goal, w
    try:
        size = int(size_entry.get())  # Read maze size from input
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter a valid integer value for maze size.")
        return

    if size <= 0:
        messagebox.showerror("Invalid Input", "Please enter a positive value for maze size.")
        return

    cols = rows = size
    w = width // cols  # cell size
    grid.clear()
    stack.clear()

    start_time = time.time()  # Start measuring time

    # Calculate the offset to center the maze
    maze_width = cols * w
    maze_height = rows * w
    x_offset = (width - maze_width) // 2
    y_offset = (height - maze_height) // 2

    class Cell:
        def __init__(self, i, j):
            self.i, self.j = i, j
            self.walls = [True, True, True, True]
            self.visited = False
            self.f_score = float('inf')  # Default f_score
            self.g_score = float('inf')
            self.parent = None

        def __lt__(self, other):
            return self.f_score < other.f_score

        def show(self, surface, is_start=False, is_goal=False):
            x = self.i * w + x_offset
            y = self.j * w + y_offset

            # Draw start and goal points
            if is_start:
                pygame.draw.rect(surface, (0, 255, 0), (x, y, w, w))  # Green for start
            elif is_goal:
                pygame.draw.rect(surface, (255, 0, 0), (x, y, w, w))  # Red for goal
            elif self.visited:
                pygame.draw.rect(surface, (255, 255, 255), (x, y, w, w))  # White for visited cells

            # Draw walls
            if self.walls[0]:
                pygame.draw.line(surface, (0, 0, 0), (x, y), (x + w, y))  # Top wall
            if self.walls[1]:
                pygame.draw.line(surface, (0, 0, 0), (x + w, y), (x + w, y + w))  # Right wall
            if self.walls[2]:
                pygame.draw.line(surface, (0, 0, 0), (x + w, y + w), (x, y + w))  # Bottom wall
            if self.walls[3]:
                pygame.draw.line(surface, (0, 0, 0), (x, y + w), (x, y))  # Left wall

        def highlight(self, surface):
            x = self.i * w + x_offset
            y = self.j * w + y_offset
            pygame.draw.rect(surface, (90, 190, 190), (x, y, w, w))

        def check_neighbors(self):
            neighbors = []
            i, j = self.i, self.j

            directions = [
                (0, -1),  # top
                (1, 0),  # right
                (0, 1),  # bottom
                (-1, 0)  # left
            ]
            random.shuffle(directions)  # Randomize neighbor checking order

            for di, dj in directions:
                ni, nj = i + di, j + dj
                idx = index(ni, nj)  # Using index function to get cell index
                if idx is not None:
                    neighbor = grid[idx]
                    if not neighbor.visited:
                        neighbors.append(neighbor)

            return random.choice(neighbors) if neighbors else None

    def index(i, j):
        if i < 0 or j < 0 or i >= cols or j >= rows:
            return None
        return i + j * cols

    def remove_walls(a, b):
        x = a.i - b.i
        if x == 1:
            a.walls[3] = False
            b.walls[1] = False
        elif x == -1:
            a.walls[1] = False
            b.walls[3] = False
        y = a.j - b.j
        if y == 1:
            a.walls[0] = False
            b.walls[2] = False
        elif y == -1:
            a.walls[2] = False
            b.walls[0] = False

    for j in range(rows):
        for i in range(cols):
            grid.append(Cell(i, j))

    # Start from a random border cell
    border_cells = [cell for cell in grid if cell.i == 0 or cell.j == 0 or cell.i == cols - 1 or cell.j == rows - 1]
    current = random.choice(border_cells)

    def step():
        global current, start, goal
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        surface = pygame.Surface((width, height))
        surface.fill((0, 0, 0))

        # Draw all cells
        for cell in grid:
            cell.show(surface, is_start=(cell == start), is_goal=(cell == goal))

        current.visited = True
        current.highlight(surface)
        next_cell = current.check_neighbors()
        if next_cell:
            next_cell.visited = True
            stack.append(current)
            remove_walls(current, next_cell)
            current = next_cell
        elif stack:
            current = stack.pop()

        # Calculate and update execution time
        end_time = time.time()
        execution_time = round(end_time - start_time, 2)
        execution_time_label.config(text=f"Execution Time: {execution_time}s")

        # Update the canvas with the maze
        img_data = pygame.image.tostring(surface, "RGB")
        img = Image.frombytes("RGB", (width, height), img_data)
        img_tk = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.img = img_tk  # Keep reference to avoid garbage collection

        if stack or not all(cell.visited for cell in grid):
            root.after(50, step)  # Continue maze generation
        else:
            set_start_and_goal()
            messagebox.showinfo("Maze Generation", "Maze generation complete!")

    def set_start_and_goal():
        """Set the start and goal points after maze generation."""
        global start, goal
        border_cells = [cell for cell in grid if cell.i == 0 or cell.j == 0 or cell.i == cols - 1 or cell.j == rows - 1]

        # Select a random start point from the border cells
        start = random.choice(border_cells)

        # Find a goal point far from the start point
        # Ensure that the goal is not too close to the start point (e.g. at least 5 cells away)
        min_distance = 5  # Minimum distance between start and goal
        valid_goal_cells = []

        for cell in border_cells:
            if cell != start:
                distance = abs(cell.i - start.i) + abs(cell.j - start.j)
                if distance >= min_distance:
                    valid_goal_cells.append(cell)

        if valid_goal_cells:
            goal = random.choice(valid_goal_cells)
        else:
            # If no valid goal cells, fall back to selecting randomly
            goal = random.choice([cell for cell in border_cells if cell != start])

        # Redraw the maze with start and goal points
        surface = pygame.Surface((width, height))
        surface.fill((0, 0, 0))

        for cell in grid:
            cell.show(surface, is_start=(cell == start), is_goal=(cell == goal))

        img_data = pygame.image.tostring(surface, "RGB")
        img = Image.frombytes("RGB", (width, height), img_data)
        img_tk = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.img = img_tk  # Keep reference to avoid garbage collection

    step()


def solve_maze_bfs():
    global start, goal
    visited = set()
    queue = deque([(start, [])])  # Queue holds tuples of (current_cell, path_to_here)
    visited.add(start)

    start_time = time.time()
    solving = True  # Flag to track if solving is still in progress

    def index(i, j):
        """Return the index of the cell based on row, column."""
        if i < 0 or j < 0 or i >= cols or j >= rows:
            return None
        return i + j * cols

    surface = pygame.Surface((width, height))
    surface.fill((0, 0, 0))

    def step():
        nonlocal solving  # Track solving status
        if not queue:
            messagebox.showinfo("Maze Solved", "No path found!")
            return

        current, path = queue.popleft()

        if current == goal:
            solving = False  # Mark as solved, so we clear the blue lines

        # Clear surface and redraw grid (except blue lines if still solving)
        surface.fill((0, 0, 0))
        for cell in grid:
            cell.show(surface, is_start=(cell == start), is_goal=(cell == goal))

        if solving:
            # Draw solving process with blue lines
            for i in range(len(path) - 1):
                x1 = path[i].i * w + x_offset + w // 2
                y1 = path[i].j * w + y_offset + w // 2
                x2 = path[i + 1].i * w + x_offset + w // 2
                y2 = path[i + 1].j * w + y_offset + w // 2
                pygame.draw.line(surface, (0, 0, 255), (x1, y1), (x2, y2), 2)  # Blue for path while solving
        else:
            # If solved, draw only the final path in red
            final_path = path + [goal]  # Include goal in the final path
            for i in range(len(final_path) - 1):
                x1 = final_path[i].i * w + x_offset + w // 2
                y1 = final_path[i].j * w + y_offset + w // 2
                x2 = final_path[i + 1].i * w + x_offset + w // 2
                y2 = final_path[i + 1].j * w + y_offset + w // 2
                pygame.draw.line(surface, (255, 0, 0), (x1, y1), (x2, y2), 3)  # Red final path

            # Update the canvas with the final path and stop execution
            img_data = pygame.image.tostring(surface, "RGB")
            img = Image.frombytes("RGB", (width, height), img_data)
            img_tk = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            canvas.img = img_tk  # Keep reference to avoid garbage collection

            messagebox.showinfo("Maze Solved", "Maze solved using BFS!")
            return

        # Explore neighbors and add to the queue
        for direction, (di, dj) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
            ni, nj = current.i + di, current.j + dj
            neighbor_idx = index(ni, nj)

            if neighbor_idx is not None:
                neighbor = grid[neighbor_idx]
                if neighbor not in visited and not current.walls[direction]:
                    visited.add(neighbor)
                    new_path = path + [current]
                    queue.append((neighbor, new_path))

        # Calculate and update execution time
        end_time = time.time()
        execution_time = round(end_time - start_time, 2)
        execution_time_label.config(text=f"Execution Time: {execution_time}s")

        # Update the canvas at each step
        img_data = pygame.image.tostring(surface, "RGB")
        img = Image.frombytes("RGB", (width, height), img_data)
        img_tk = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.img = img_tk  # Keep reference to avoid garbage collection

        root.after(50, step)  # Continue to the next step

    step()


def solve_maze_dfs():
    global start, goal
    visited = set()
    stack = [(start, [])]  # Stack holds tuples of (current_cell, path_to_here)
    visited.add(start)

    start_time = time.time()

    def index(i, j):
        """Return the index of the cell based on row, column."""
        if i < 0 or j < 0 or i >= cols or j >= rows:
            return None
        return i + j * cols

    surface = pygame.Surface((width, height))
    surface.fill((0, 0, 0))

    def step():
        if not stack:
            messagebox.showinfo("Maze Solved", "No path found!")
            return

        current, path = stack.pop()  # Pop from the stack (DFS behavior)

        if current == goal:
            path.append(goal)  # Ensure the final step to goal is included
            for i in range(len(path) - 1):
                x1 = path[i].i * w + x_offset + w // 2
                y1 = path[i].j * w + y_offset + w // 2
                x2 = path[i + 1].i * w + x_offset + w // 2
                y2 = path[i + 1].j * w + y_offset + w // 2
                pygame.draw.line(surface, (255, 0, 0), (x1, y1), (x2, y2), 3)

            # Update the canvas with the final path
            img_data = pygame.image.tostring(surface, "RGB")
            img = Image.frombytes("RGB", (width, height), img_data)
            img_tk = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            canvas.img = img_tk  # Keep reference to avoid garbage collection

            messagebox.showinfo("Maze Solved", "Maze solved using DFS!")
            return

        # Explore neighbors (same order as BFS)
        for direction, (di, dj) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
            ni, nj = current.i + di, current.j + dj
            neighbor_idx = index(ni, nj)

            if neighbor_idx is not None:
                neighbor = grid[neighbor_idx]
                if neighbor not in visited and not current.walls[direction]:
                    visited.add(neighbor)
                    new_path = path + [current]
                    stack.append((neighbor, new_path))  # Push to stack

        # Calculate and update execution time
        end_time = time.time()
        execution_time = round(end_time - start_time, 2)
        execution_time_label.config(text=f"Execution Time: {execution_time}s")

        # Clear the surface for the next step and draw current state
        surface.fill((0, 0, 0))  # Clear the surface for the next step
        for cell in grid:
            cell.show(surface, is_start=(cell == start), is_goal=(cell == goal))

        # Draw the path so far in blue (line connecting the current path cells)
        for i in range(len(path) - 1):
            x1 = path[i].i * w + x_offset + w // 2
            y1 = path[i].j * w + y_offset + w // 2
            x2 = path[i + 1].i * w + x_offset + w // 2
            y2 = path[i + 1].j * w + y_offset + w // 2
            pygame.draw.line(surface, (0, 0, 255), (x1, y1), (x2, y2), 2)  # Blue for path

        # Update the canvas at each step
        img_data = pygame.image.tostring(surface, "RGB")
        img = Image.frombytes("RGB", (width, height), img_data)
        img_tk = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.img = img_tk  # Keep reference to avoid garbage collection

        root.after(50, step)  # Continue to the next step

    step()


def solve_maze_astar():
    global start, goal
    visited = set()
    came_from = {}
    g_score = {start: 0}
    
    def heuristic(a, b):
        return abs(a.i - b.i) + abs(a.j - b.j)

    f_score = {start: heuristic(start, goal)}
    open_set = []
    heapq.heappush(open_set, (f_score[start], start))

    start_time = time.time()
    surface = pygame.Surface((width, height))
    surface.fill((0, 0, 0))

    def index(i, j):
        if i < 0 or j < 0 or i >= cols or j >= rows:
            return None
        return i + j * cols

    def reconstruct_path(came_from, current):
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        return path

    def step():
        if not open_set:
            messagebox.showinfo("Maze Solved", "No path found!")
            return

        _, current = heapq.heappop(open_set)
        
        if current == goal:
            final_path = reconstruct_path(came_from, current)
            for i in range(len(final_path) - 1):
                x1 = final_path[i].i * w + x_offset + w // 2
                y1 = final_path[i].j * w + y_offset + w // 2
                x2 = final_path[i + 1].i * w + x_offset + w // 2
                y2 = final_path[i + 1].j * w + y_offset + w // 2
                pygame.draw.line(surface, (255, 0, 0), (x1, y1), (x2, y2), 3)
            
            img_data = pygame.image.tostring(surface, "RGB")
            img = Image.frombytes("RGB", (width, height), img_data)
            img_tk = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            canvas.img = img_tk
            
            messagebox.showinfo("Maze Solved", "Maze solved using A*!")
            return

        surface.fill((0, 0, 0))
        for cell in grid:
            cell.show(surface, is_start=(cell == start), is_goal=(cell == goal))

        path = reconstruct_path(came_from, current)
        for i in range(len(path) - 1):
            x1 = path[i].i * w + x_offset + w // 2
            y1 = path[i].j * w + y_offset + w // 2
            x2 = path[i + 1].i * w + x_offset + w // 2
            y2 = path[i + 1].j * w + y_offset + w // 2
            pygame.draw.line(surface, (0, 0, 255), (x1, y1), (x2, y2), 2)
        
        for direction, (di, dj) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
            ni, nj = current.i + di, current.j + dj
            neighbor_idx = index(ni, nj)
            
            if neighbor_idx is not None:
                neighbor = grid[neighbor_idx]
                if neighbor not in visited and not current.walls[direction]:
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        end_time = time.time()
        execution_time = round(end_time - start_time, 2)
        execution_time_label.config(text=f"Execution Time: {execution_time}s")
        
        img_data = pygame.image.tostring(surface, "RGB")
        img = Image.frombytes("RGB", (width, height), img_data)
        img_tk = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.img = img_tk
        
        root.after(50, step)
    
    step()


def solve_maze_selected():
    """ Calls the selected maze-solving algorithm."""
    if algorithm_var.get() == "BFS":
        solve_maze_bfs()  # Calls BFS
    elif algorithm_var.get() == "A*":
        solve_maze_astar()
    else:
        solve_maze_dfs()  # Calls DFS

        
# create tkinter window
root = tk.Tk()
root.title("Maze Solver")

# Create a top frame with a colored background
top_frame = tk.Frame(root, bg="#d3d3d3", padx=10, pady=10)  # Light gray background
top_frame.pack(side=tk.TOP, fill=tk.X)

# create time label
execution_time_label = tk.Label(root, text="Execution Time: 0.0s", font=('Arial', 12))
execution_time_label.pack(pady=10)

# Maze size input
size_label = tk.Label(top_frame, text="Maze Size:", bg="#d3d3d3")
size_label.pack(side=tk.LEFT, padx=5)
size_entry = tk.Entry(top_frame, width=5)
size_entry.pack(side=tk.LEFT, padx=5)

# Generate maze button
generate_button = tk.Button(top_frame, text="Generate Maze", command=generate_maze)
generate_button.pack(side=tk.LEFT, padx=5)

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