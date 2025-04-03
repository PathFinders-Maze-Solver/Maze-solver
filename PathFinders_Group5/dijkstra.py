import heapq
import time
import pygame
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

def solve_maze_dijkstra(start, goal, grid, index, canvas, execution_time_label, root, width, height, w, x_offset, y_offset):
    visited = set()
    came_from = {}
    g_score = {cell: float('inf') for cell in grid}
    g_score[start] = 0

    # Priority queue with (cost, cell)
    open_set = []
    heapq.heappush(open_set, (0, start))

    start_time = time.perf_counter()
    surface = pygame.Surface((width, height))
    surface.fill((255, 255, 255))

    def reconstruct_path(came_from, current):
        """Reconstructs the shortest path from the start to the goal."""
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        return path

    def step():
        """One step of Dijkstra's algorithm."""
        if not open_set:
            messagebox.showinfo("Maze Solved", "No path found!")
            return

        # Get the node with the lowest cost
        _, current = heapq.heappop(open_set)

        # Explore neighbors and update cost
        for direction, (di, dj) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
            ni, nj = current.i + di, current.j + dj
            neighbor_idx = index(ni, nj)
            if neighbor_idx is not None:
                neighbor = grid[neighbor_idx]
                if not current.walls[direction]:
                    tentative_g_score = g_score[current] + 1  # Uniform cost for every move
                    if tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        heapq.heappush(open_set, (g_score[neighbor], neighbor))

        # If goal reached, clear any blue lines and show only the final red path
        if current == goal:
            surface.fill((255, 255, 255))
            for cell in grid:
                cell.show(surface, is_start=(cell == start), is_goal=(cell == goal))
            final_path = reconstruct_path(came_from, current)
            for i in range(len(final_path) - 1):
                x1 = final_path[i].i * w + x_offset + w // 2
                y1 = final_path[i].j * w + y_offset + w // 2
                x2 = final_path[i + 1].i * w + x_offset + w // 2
                y2 = final_path[i + 1].j * w + y_offset + w // 2
                pygame.draw.line(surface, (255, 0, 0), (x1, y1), (x2, y2), 3)  # Red for final path

            img_data = pygame.image.tostring(surface, "RGB")
            img = Image.frombytes("RGB", (width, height), img_data)
            img_tk = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            canvas.img = img_tk

            messagebox.showinfo("Maze Solved", "Maze solved using Dijkstra's Algorithm!")
            return

        # Clear surface and redraw grid
        surface.fill((255, 255, 255))
        for cell in grid:
            cell.show(surface, is_start=(cell == start), is_goal=(cell == goal))

        # Draw the current search progress in blue
        path = reconstruct_path(came_from, current)
        for i in range(len(path) - 1):
            x1 = path[i].i * w + x_offset + w // 2
            y1 = path[i].j * w + y_offset + w // 2
            x2 = path[i + 1].i * w + x_offset + w // 2
            y2 = path[i + 1].j * w + y_offset + w // 2
            pygame.draw.line(surface, (0, 0, 255), (x1, y1), (x2, y2), 2)  # Blue for search path

        # Update execution time
        execution_time_label.config(text=f"Execution Time: {round(time.perf_counter() - start_time, 2)}ms")

        # Update the canvas
        img_data = pygame.image.tostring(surface, "RGB")
        img = Image.frombytes("RGB", (width, height), img_data)
        img_tk = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.img = img_tk

        root.after(1, step)  # Continue to the next step

    step()
