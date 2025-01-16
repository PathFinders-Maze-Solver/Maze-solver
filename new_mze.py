import pygame
import sys
import random
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

pygame.init()

# Global variables for maze state
cols, rows, grid, stack, current = 0, 0, [], [], None
w = 0

def generate_maze():
    global cols, rows, grid, stack, current, w
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

    class Cell:
        def __init__(self, i, j):
            self.i, self.j = i, j
            self.walls = [True, True, True, True]
            self.visited = False

        def show(self, surface):
            x = self.i * w
            y = self.j * w
            if self.visited:
                pygame.draw.rect(surface, (255, 255, 255), (x, y, w, w))
            if self.walls[0]:
                pygame.draw.line(surface, (0, 0, 0), (x, y), (x + w, y))
            if self.walls[1]:
                pygame.draw.line(surface, (0, 0, 0), (x + w, y), (x + w, y + w))
            if self.walls[2]:
                pygame.draw.line(surface, (0, 0, 0), (x + w, y + w), (x, y + w))
            if self.walls[3]:
                pygame.draw.line(surface, (0, 0, 0), (x, y + w), (x, y))

        def highlight(self, surface):
            x = self.i * w
            y = self.j * w
            pygame.draw.rect(surface, (90, 190, 190), (x, y, w, w))

        def checkNeighbors(self):
            neighbors = []
            i, j = self.i, self.j
            if index(i, j - 1):
                top = grid[index(i, j - 1)]
                if not top.visited:
                    neighbors.append(top)
            if index(i + 1, j):
                right = grid[index(i + 1, j)]
                if not right.visited:
                    neighbors.append(right)
            if index(i - 1, j):
                left = grid[index(i - 1, j)]
                if not left.visited:
                    neighbors.append(left)
            if index(i, j + 1):
                bottom = grid[index(i, j + 1)]
                if not bottom.visited:
                    neighbors.append(bottom)

            if neighbors:
                return random.choice(neighbors)
            else:
                return None

    def index(i, j):
        if i < 0 or j < 0 or i >= cols or j >= rows:
            return None
        return i + j * cols

    def removeWalls(a, b):
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

    current = grid[0]

    def step():
        global current
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        surface = pygame.Surface((600, 600))
        surface.fill((0, 0, 0))

        for cell in grid:
            cell.show(surface)

        current.visited = True
        current.highlight(surface)
        nextcell = current.checkNeighbors()
        if nextcell:
            nextcell.visited = True
            stack.append(current)
            removeWalls(current, nextcell)
            current = nextcell
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
            messagebox.showinfo("Maze Generation", "Maze generation complete!")

    step()

# Initialize Tkinter window
root = tk.Tk()
root.title("Maze Generator")
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
