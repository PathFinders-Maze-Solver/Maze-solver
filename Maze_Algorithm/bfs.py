import time
import pygame
from collections import deque
from tkinter import messagebox
from PIL import Image, ImageTk

def solve_maze_bfs(start, goal, grid, index, update_gui, canvas, execution_time_label, root, width, height):
    visited = set()
    queue = deque([(start, [])])  # Queue holds tuples of (current_cell, path_to_here)
    visited.add(start)

    start_time = time.time()
    solving = True  # Flag to track if solving is still in progress

    surface = pygame.Surface((width, height))
    surface.fill((255, 255, 255))

    def step():
        nonlocal solving  # Track solving status
        if not queue:
            messagebox.showinfo("Maze Solved using BFS", "No path found!")
            return

        current, path = queue.popleft()

        if current == goal:
            solving = False  # Mark as solved, so we clear the blue lines

        # Update the GUI at each step
        update_gui(path, solving, surface)

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
        canvas.create_image(0, 0, anchor="nw", image=img_tk)
        canvas.img = img_tk  # Keep reference to avoid garbage collection

        # Continue solving until the goal is reached
        if solving:
            root.after(50, step)  # Continue to the next step

    # Start the first step
    step()
