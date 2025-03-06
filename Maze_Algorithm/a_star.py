import time
import heapq
import pygame
from tkinter import messagebox
from PIL import Image, ImageTk

def heuristic(cell, goal):
    """Calculate the Manhattan distance heuristic."""
    return abs(cell.i - goal.i) + abs(cell.j - goal.j)

def solve_maze_a_star(start, goal, grid, index, update_gui, canvas, execution_time_label, root, width, height):
    open_list = []  # Priority queue (min-heap) for A*
    heapq.heappush(open_list, (0, start, []))  # (f, cell, path_to_here)
    came_from = {}  # To reconstruct the path
    g_score = {start: 0}  # Cost from start to the current cell
    f_score = {start: heuristic(start, goal)}  # Estimated cost from start to goal through current cell

    visited = set()

    start_time = time.time()
    solving = True  # Flag to track if solving is still in progress

    surface = pygame.Surface((width, height))
    surface.fill((255, 255, 255))

    def step():
        nonlocal solving  # Track solving status
        if not open_list:
            messagebox.showinfo("Maze Solved using A_Star", "No path found!")
            return

        # Get the node with the lowest f_score
        _, current, path = heapq.heappop(open_list)

        if current == goal:
            solving = False  # Mark as solved, so we clear the blue lines

        # Update the GUI at each step
        update_gui(path, solving, surface)

        # Explore neighbors and add to the open list
        for direction, (di, dj) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
            ni, nj = current.i + di, current.j + dj
            neighbor_idx = index(ni, nj)

            if neighbor_idx is not None:
                neighbor = grid[neighbor_idx]
                if neighbor not in visited and not current.walls[direction]:
                    tentative_g_score = g_score[current] + 1  # Assuming all moves cost 1

                    # If this path to the neighbor is better, update the scores and path
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_list, (f_score[neighbor], neighbor, path + [current]))

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
