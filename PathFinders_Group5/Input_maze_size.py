import tkinter as tk
from tkinter import messagebox
import random
import time
import pygame
import sys
from PIL import Image, ImageTk

from a_star import solve_maze_a_star
from dijkstra import solve_maze_dijkstra
from bfs import solve_maze_bfs
from dfs import solve_maze_dfs

# Initialize other variables and functions as needed
cols, rows, grid, stack, current, start, goal = 0, 0, [], [], None, None, None
w = 0
x_offset = 0
y_offset = 0
width, height = 530, 530

# Add a setup function to initialize the content in the provided frame
def setup(parent_frame):
    global root, canvas, execution_time_label, size_entry, algorithm_var, generate_button, solve_button
    


    def generate_maze():
        global cols, rows, grid, stack, current, start, goal, w
        try:
            size = int(size_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid integer value for maze size.")
            return

        if size <= 0:
            messagebox.showerror("Invalid Input", "Please enter a positive value for maze size.")
            return

        cols = rows = size
        w = width // cols
        grid.clear()
        stack.clear()

        start_time = time.time()

        maze_width = cols * w
        maze_height = rows * w
        x_offset = (width - maze_width) // 2
        y_offset = (height - maze_height) // 2

        class Cell:
            def __init__(self, i, j):
                self.i, self.j = i, j
                self.walls = [True, True, True, True]
                self.visited = False
                self.parent = None
                self.rank = 0
                self.f_score = float('inf')
                self.g_score = float('inf')

            def __lt__(self, other):
                # Comparison method for heapq to use when comparing cells
                return self.f_score < other.f_score

            def __eq__(self, other):
                # Check if two cells are at the same location
                return self.i == other.i and self.j == other.j

            def __hash__(self):
                # Allow cells to be used in sets and as dictionary keys
                return hash((self.i, self.j))

            def show(self, surface, is_start=False, is_goal=False):
                x = self.i * w + x_offset
                y = self.j * w + y_offset

                # Fill cell with white
                pygame.draw.rect(surface, (255, 255, 255), (x, y, w, w))

                # Draw start and goal points
                if is_start:
                    pygame.draw.rect(surface, (0, 255, 0), (x, y, w, w))  # Green for start
                elif is_goal:
                    pygame.draw.rect(surface, (255, 0, 0), (x, y, w, w))  # Red for goal

                # Draw black walls
                if self.walls[0]:
                    pygame.draw.line(surface, (0, 0, 0), (x, y), (x + w, y), 2)  # Top wall
                if self.walls[1]:
                    pygame.draw.line(surface, (0, 0, 0), (x + w, y), (x + w, y + w), 2)  # Right wall
                if self.walls[2]:
                    pygame.draw.line(surface, (0, 0, 0), (x + w, y + w), (x, y + w), 2)  # Bottom wall
                if self.walls[3]:
                    pygame.draw.line(surface, (0, 0, 0), (x, y + w), (x, y), 2)  # Left wall

            def highlight(self, surface):
                x = self.i * w + x_offset
                y = self.j * w + y_offset
                pygame.draw.rect(surface, (200, 200, 200), (x, y, w, w))

        def index(i, j):
            if i < 0 or j < 0 or i >= cols or j >= rows:
                return None
            return i + j * cols

        def find(cell):
            if cell.parent is None:
                return cell
            cell.parent = find(cell.parent)
            return cell.parent

        def union(cell1, cell2):
            root1 = find(cell1)
            root2 = find(cell2)

            if root1 == root2:
                return False

            if root1.rank < root2.rank:
                root1.parent = root2
            elif root1.rank > root2.rank:
                root2.parent = root1
            else:
                root2.parent = root1
                root1.rank += 1
            return True

        # Create grid of cells
        for j in range(rows):
            for i in range(cols):
                grid.append(Cell(i, j))

        # Generate all possible walls
        walls = []
        for j in range(rows):
            for i in range(cols):
                cell_index = index(i, j)
                current_cell = grid[cell_index]

                # Check right wall
                if i < cols - 1:
                    right_index = index(i + 1, j)
                    if right_index is not None:
                        walls.append((current_cell, grid[right_index], 1))

                # Check bottom wall
                if j < rows - 1:
                    bottom_index = index(i, j + 1)
                    if bottom_index is not None:
                        walls.append((current_cell, grid[bottom_index], 2))

        # Shuffle walls to randomize maze generation
        random.shuffle(walls)

        # Kruskal's algorithm to generate maze
        def step():
            nonlocal walls
            surface = pygame.Surface((width, height))
            surface.fill((255, 255, 255))  # White background

            # Draw all cells
            for cell in grid:
                cell.show(surface)

            # If walls remain, remove a wall
            if walls:
                cell1, cell2, wall_direction = walls.pop()

                if union(cell1, cell2):
                    # Remove walls between connected cells
                    if wall_direction == 1:  # Right wall
                        cell1.walls[1] = False
                        cell2.walls[3] = False
                    else:  # Bottom wall
                        cell1.walls[2] = False
                        cell2.walls[0] = False

                    # Highlight current cells being processed
                    cell1.highlight(surface)
                    cell2.highlight(surface)

            # Calculate and update execution time
            end_time = time.time()
            execution_time = round(end_time - start_time, 2)
            execution_time_label.config(text=f"Execution Time: {execution_time}s")

            # Update the canvas with the maze
            img_data = pygame.image.tostring(surface, "RGB")
            img = Image.frombytes("RGB", (width, height), img_data)
            img_tk = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            canvas.img = img_tk

            if walls:
                root.after(1, step)  # Continue maze generation
            else:
                set_start_and_goal()
                messagebox.showinfo("Maze Generation", "Maze generation complete!")

        def set_start_and_goal():
            global start, goal
            border_cells = [cell for cell in grid if cell.i == 0 or cell.j == 0 or cell.i == cols - 1 or cell.j == rows - 1]

            # Select a random start point from the border cells
            start = random.choice(border_cells)

            # Find a goal point far from the start point
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
                goal = random.choice([cell for cell in border_cells if cell != start])

            # Remove walls for start cell based on its border location
            if start.i == 0:  # Left border
                start.walls[3] = False
            elif start.i == cols - 1:  # Right border
                start.walls[1] = False
            elif start.j == 0:  # Top border
                start.walls[0] = False
            elif start.j == rows - 1:  # Bottom border
                start.walls[2] = False

            # Remove walls for goal cell based on its border location
            if goal.i == 0:  # Left border
                goal.walls[3] = False
            elif goal.i == cols - 1:  # Right border
                goal.walls[1] = False
            elif goal.j == 0:  # Top border
                goal.walls[0] = False
            elif goal.j == rows - 1:  # Bottom border
                goal.walls[2] = False

            # Redraw the maze with start and goal points
            surface = pygame.Surface((width, height))
            surface.fill((255, 255, 255))

            for cell in grid:
                cell.show(surface, is_start=(cell == start), is_goal=(cell == goal))

            img_data = pygame.image.tostring(surface, "RGB")
            img = Image.frombytes("RGB", (width, height), img_data)
            img_tk = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            canvas.img = img_tk

        step()


    def update_gui(path, solving, surface):
        surface.fill((255, 255, 255))  # Clear the surface
        for cell in grid:
            cell.show(surface, is_start=(cell == start), is_goal=(cell == goal))

        # Display the current state of the maze as BFS solves it
        if solving:
            # Draw real-time solving process with blue lines
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

            # Show the solved path on the canvas
            img_data = pygame.image.tostring(surface, "RGB")
            img = Image.frombytes("RGB", (width, height), img_data)
            img_tk = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            canvas.img = img_tk  # Keep reference to avoid garbage collection

            messagebox.showinfo("Maze Solved", "Maze solved")
            return  # Exit after solving

    def index(i, j):
        """Return the index of the cell based on row, column."""
        if i < 0 or j < 0 or i >= cols or j >= rows:
            return None
        return i + j * cols


    def solve_maze_selected():
        """ Calls the selected maze-solving algorithm."""
        if algorithm_var.get() == "BFS":
            solve_maze_bfs(start, goal, grid, index, update_gui, canvas, execution_time_label, root, width, height)
        elif algorithm_var.get() == "A*":
            solve_maze_a_star(start, goal, grid, index, update_gui, canvas, execution_time_label, root, width, height)
        elif algorithm_var.get() == "DFS":
            solve_maze_dfs(start, goal, grid, index, update_gui, canvas, execution_time_label, root, width, height)
        elif algorithm_var.get() == "Dijkstra":
            solve_maze_dijkstra(start, goal, grid, index, canvas, execution_time_label, root, width, height,w,x_offset,y_offset)



    def reset_maze():
        # Create a new surface and fill with white (clearing any previous drawings)
        surface = pygame.Surface((width, height))
        surface.fill((255, 255, 255))

        # Redraw every cell (without any solution path)
        for cell in grid:
            cell.show(surface, is_start=(cell == start), is_goal=(cell == goal))

        # Update the canvas with the redrawn maze
        img_data = pygame.image.tostring(surface, "RGB")
        img = Image.frombytes("RGB", (width, height), img_data)
        img_tk = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.img = img_tk  # Keep a reference

        # Optionally, reset the execution time label
        execution_time_label.config(text="Execution Time: 0s")

    # Function to clear the maze
    def clear_maze():
        global cols, rows, grid, stack, current, start, goal
        cols, rows = 0, 0
        grid.clear()
        stack.clear()
        start, goal = None, None
        canvas.delete("all")  # Clear the canvas
        execution_time_label.config(text="Execution Time: 0s")  # clear execution time label
        size_entry.delete(0, tk.END)  # Clear the maze size input



    # Replace the root window with the provided parent frame
    root = parent_frame

    # Initialize GUI components
    top_frame = tk.Frame(root, bg="#d3d3d3", padx=10, pady=10)
    top_frame.pack(side=tk.TOP, fill=tk.X)

    execution_time_label = tk.Label(root, text="Execution Time: 0.0s", font=('Arial', 12))
    execution_time_label.pack(pady=10)

    size_label = tk.Label(top_frame, text="Maze Size:", bg="#d3d3d3")
    size_label.pack(side=tk.LEFT, padx=5)
    size_entry = tk.Entry(top_frame, width=5)
    size_entry.pack(side=tk.LEFT, padx=5)

    generate_button = tk.Button(top_frame, text="Generate Maze", command=generate_maze)
    generate_button.pack(side=tk.LEFT, padx=5)
    clear_button = tk.Button(top_frame, text="Clear Maze", command=clear_maze)
    clear_button.pack(side=tk.LEFT, padx=5)
    reset_button = tk.Button(top_frame, text="Reset Maze", command=reset_maze)
    reset_button.pack(side=tk.LEFT, padx=5)

    solve_button = tk.Button(top_frame, text="Solve Maze", command=solve_maze_selected)
    solve_button.pack(side=tk.RIGHT, padx=10)

    algo_frame = tk.Frame(top_frame, bg="#d3d3d3")
    algo_frame.pack(side=tk.RIGHT)

    algorithm_var = tk.StringVar(value="BFS")
    astar_radio = tk.Radiobutton(algo_frame, text="A*", variable=algorithm_var, value="A*", bg="#d3d3d3")
    astar_radio.pack(side=tk.RIGHT, padx=5)
    dfs_radio = tk.Radiobutton(algo_frame, text="DFS", variable=algorithm_var, value="DFS", bg="#d3d3d3")
    dfs_radio.pack(side=tk.RIGHT, padx=5)
    bfs_radio = tk.Radiobutton(algo_frame, text="BFS", variable=algorithm_var, value="BFS", bg="#d3d3d3")
    bfs_radio.pack(side=tk.RIGHT, padx=5)
    dijkstra_radio = tk.Radiobutton(algo_frame, text="Dijkstra", variable=algorithm_var, value="Dijkstra", bg="#d3d3d3")
    dijkstra_radio.pack(side=tk.RIGHT, padx=5)

    algo_label = tk.Label(top_frame, text="Algorithm:", bg="#d3d3d3")
    algo_label.pack(side=tk.RIGHT, padx=5)

    canvas = tk.Canvas(root, width=600, height=600)
    canvas.pack()

    