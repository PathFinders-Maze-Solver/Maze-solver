import time
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
    w = 600 // cols  # cell size
    grid.clear()
    stack.clear()

    start_time = time.time()
    

    # Calculate the offset to center the maze
    maze_width = cols * w
    maze_height = rows * w
    x_offset = (600 - maze_width) // 2
    y_offset = (600 - maze_height) // 2

    class Cell:
        def __init__(self, i, j):
            self.i, self.j = i, j
            self.walls = [True, True, True, True]
            self.visited = False

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

        surface = pygame.Surface((600, 600))
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

        img_data = pygame.image.tostring(surface, "RGB")
        img = Image.frombytes("RGB", (600, 600), img_data)
        img_tk = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.img = img_tk  # Keep reference to avoid garbage collection

        if stack or not all(cell.visited for cell in grid):
            root.after(50, step)  # Continue maze generation
        else:
            set_start_and_goal()
            end_time = time.time()  # End the timer
            execution_time = end_time - start_time  # Calculate execution time
            execution_time_label.config(text=f"Execution Time: {execution_time}s")  # Update the label
            messagebox.showinfo("Maze Generation", f"Maze generation complete! Time taken: {execution_time:.2f}s")
            

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
        surface = pygame.Surface((600, 600))
        surface.fill((0, 0, 0))

        for cell in grid:
            cell.show(surface, is_start=(cell == start), is_goal=(cell == goal))

        img_data = pygame.image.tostring(surface, "RGB")
        img = Image.frombytes("RGB", (600, 600), img_data)
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

    def index(i, j):
        """Return the index of the cell based on row, column."""
        if i < 0 or j < 0 or i >= cols or j >= rows:
            return None
        return i + j * cols

    surface = pygame.Surface((600, 600))
    surface.fill((0, 0, 0))

    def step():
        if not queue:
            messagebox.showinfo("Maze Solved", "No path found!")
            return

        current, path = queue.popleft()

        if current == goal:
            # If goal is reached, draw the entire path in red
            for i in range(len(path) - 1):
                x1 = path[i].i * w + x_offset + w // 2
                y1 = path[i].j * w + y_offset + w // 2
                x2 = path[i + 1].i * w + x_offset + w // 2
                y2 = path[i + 1].j * w + y_offset + w // 2
                pygame.draw.line(surface, (255, 0, 0), (x1, y1), (x2, y2), 3)

            # Update the canvas with the final path
            img_data = pygame.image.tostring(surface, "RGB")
            img = Image.frombytes("RGB", (600, 600), img_data)
            img_tk = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            canvas.img = img_tk  # Keep reference to avoid garbage collection

            messagebox.showinfo("Maze Solved", "Maze solved using BFS!")
            return

        # Explore neighbors
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
        img = Image.frombytes("RGB", (600, 600), img_data)
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

    surface = pygame.Surface((600, 600))
    surface.fill((0, 0, 0))

    def step():
        if not stack:
            messagebox.showinfo("Maze Solved", "No path found!")
            return

        current, path = stack.pop()  # Pop from the stack (DFS behavior)

        if current == goal:
            # If goal is reached, draw the path with red lines
            for i in range(len(path) - 1):
                x1 = path[i].i * w + x_offset + w // 2
                y1 = path[i].j * w + y_offset + w // 2
                x2 = path[i + 1].i * w + x_offset + w // 2
                y2 = path[i + 1].j * w + y_offset + w // 2
                pygame.draw.line(surface, (255, 0, 0), (x1, y1), (x2, y2), 3)

            # Update the canvas with the final path
            img_data = pygame.image.tostring(surface, "RGB")
            img = Image.frombytes("RGB", (600, 600), img_data)
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
        img = Image.frombytes("RGB", (600, 600), img_data)
        img_tk = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.img = img_tk  # Keep reference to avoid garbage collection

        root.after(50, step)  # Continue to the next step

        

    step()



root = tk.Tk()
root.title("Maze Solver")

canvas = tk.Canvas(root, width=600, height=600)
canvas.pack()

execution_time_label = tk.Label(root, text="Execution Time: 0.0s", font=('Arial', 12))
execution_time_label.pack(pady=10)

# Input size entry
size_label = tk.Label(root, text="Maze Size (e.g., 10):")
size_label.pack()
size_entry = tk.Entry(root)
size_entry.pack()

generate_button = tk.Button(root, text="Generate Maze", command=generate_maze)
generate_button.pack()

# Radio buttons to select algorithm
algorithm_var = tk.StringVar(value="BFS")  # Default selection is BFS

bfs_radio = tk.Radiobutton(root, text="BFS", variable=algorithm_var, value="BFS")
bfs_radio.pack()

dfs_radio = tk.Radiobutton(root, text="DFS", variable=algorithm_var, value="DFS")
dfs_radio.pack()

def solve_maze_selected():
    """Calls the selected maze-solving algorithm."""
    if algorithm_var.get() == "BFS":
        solve_maze_bfs()  # Calls BFS
    else:
        solve_maze_dfs()  # Calls DFS

solve_button = tk.Button(root, text="Solve Maze", command=solve_maze_selected)
solve_button.pack()

root.mainloop()
