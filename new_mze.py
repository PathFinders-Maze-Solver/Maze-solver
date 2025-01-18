import pygame
import sys
import random
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

pygame.init()

# Global variables for maze state
cols, rows, grid, stack, current, start, goal = 0, 0, [], [], None, None, None
w = 0


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
                idx = index(ni, nj)
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


# Initialize Tkinter window
root = tk.Tk()
root.title("Maze Generator with One Goal")
root.geometry("600x650")

maze_size_label = tk.Label(root, text="Enter Maze Size (e.g. 9 for 9x9):")
maze_size_label.pack()

size_entry = tk.Entry(root)
size_entry.insert(0, "9")
size_entry.pack()

generate_button = tk.Button(root, text="Generate Maze", command=generate_maze)
generate_button.pack()

canvas = tk.Canvas(root, width=600, height=600, bg="black")
canvas.pack()

root.mainloop()
