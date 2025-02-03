import random
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os

class MazeGenerator:
    def __init__(self):
        # Constants
        self.TILE_SIZE = 32
        self.OPEN = 1
        self.CLOSED = 0
        
        # Initialize maze
        self.maze = []
        
        # Create the main window
        self.root = tk.Tk()
        self.root.title("Maze Generator")
        
        # Create controls
        self.create_controls()
        
        # Create canvas for maze display
        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(padx=10, pady=10)
        
    def create_controls(self):
        control_frame = ttk.Frame(self.root)
        control_frame.pack(padx=10, pady=5)
        
        # Width controls
        ttk.Label(control_frame, text="Dimension:").pack(side=tk.LEFT)
        self.dimension_var = tk.StringVar(value="9")
        self.dimension_entry = ttk.Entry(control_frame, textvariable=self.dimension_var, width=5)
        self.dimension_entry.pack(side=tk.LEFT, padx=5)
        
        # # Height controls
        # ttk.Label(control_frame, text="Height:").pack(side=tk.LEFT)
        # self.height_var = tk.StringVar(value="15")
        # self.height_entry = ttk.Entry(control_frame, textvariable=self.height_var, width=5)
        # self.height_entry.pack(side=tk.LEFT, padx=5)
        
        # Buttons
        ttk.Button(control_frame, text="Generate Maze", command=self.create_maze).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Save Maze", command=self.download_maze).pack(side=tk.LEFT, padx=5)

    def create_maze(self):
        # Get dimensions
        width = int(self.dimension_var.get())
        height = int(self.dimension_var.get())
        
        # Reset maze with solid blocks
        self.maze = self.create_2d_array(width, height, self.CLOSED)
        
        # Find starting position
        start_x = self.set_to_odd(random.randint(1, width-2))
        start_y = self.set_to_odd(random.randint(1, height-2))
        
        # Generate the maze
        self.dig_around(start_x, start_y)
        self.create_egresses()
        
        # Draw the maze
        self.draw_maze()

    def dig_around(self, x, y):
        self.maze[y][x] = self.OPEN
        self.draw_maze()  # Update the display after marking the current cell
        self.canvas.update()  # Force the canvas to update
        
        neighbors = [
            {"x": x-2, "y": y},    # Left
            {"x": x+2, "y": y},    # Right
            {"x": x, "y": y-2},    # Up
            {"x": x, "y": y+2}     # Down
        ]
        random.shuffle(neighbors)
        
        for neighbor in neighbors:
            self.dig_to(neighbor["x"], neighbor["y"], x, y)
        
        # Add a small delay to slow down the animation
        self.root.after(50, self.update_maze)  # Delay in milliseconds

    def dig_to(self, dest_x, dest_y, from_x, from_y):
        mid_x = (dest_x + from_x) // 2
        mid_y = (dest_y + from_y) // 2
        
        # Check if within bounds
        if not self.is_within_map(dest_x, dest_y):
            return
            
        # Check if we haven't dug here before
        if (self.maze[dest_y][dest_x] == self.CLOSED and 
            self.maze[mid_y][mid_x] == self.CLOSED):
            # Dig the path
            self.maze[dest_y][dest_x] = self.OPEN
            self.maze[mid_y][mid_x] = self.OPEN
            
            # Continue digging from new position
            self.dig_around(dest_x, dest_y)

    def is_within_map(self, x, y):
        return (0 <= x < len(self.maze[0]) and 
                0 <= y < len(self.maze))

    def create_egresses(self):
        if random.random() > 0.5:
            # Add entrance and exit left and right
            entrance_y = self.set_to_odd(random.randint(1, len(self.maze)-2))
            exit_y = self.set_to_odd(random.randint(1, len(self.maze)-2))
            
            self.maze[entrance_y][0] = self.OPEN
            self.maze[exit_y][len(self.maze[0])-1] = self.OPEN
        else:
            # Add entrance and exit top and bottom
            entrance_x = self.set_to_odd(random.randint(1, len(self.maze[0])-2))
            exit_x = self.set_to_odd(random.randint(1, len(self.maze[0])-2))
            
            self.maze[0][entrance_x] = self.OPEN
            self.maze[len(self.maze)-1][exit_x] = self.OPEN

    def draw_maze(self):
        # Calculate canvas size
        canvas_width = len(self.maze[0]) * self.TILE_SIZE
        canvas_height = len(self.maze) * self.TILE_SIZE
        
        # Configure canvas
        self.canvas.config(width=canvas_width, height=canvas_height)
        self.canvas.delete("all")
        
        # Draw each cell
        for y in range(len(self.maze)):
            for x in range(len(self.maze[0])):
                color = "white" if self.maze[y][x] == self.OPEN else "black"
                self.canvas.create_rectangle(
                    x * self.TILE_SIZE, 
                    y * self.TILE_SIZE,
                    (x + 1) * self.TILE_SIZE, 
                    (y + 1) * self.TILE_SIZE,
                    fill=color, 
                    outline="gray"
                )

    def download_maze(self):
        with open("maze.txt", "w") as f:
            for row in self.maze:
                f.write("".join(str(cell) for cell in row) + "\n")

    @staticmethod
    def create_2d_array(width, height, fill_with):
        return [[fill_with for _ in range(width)] for _ in range(height)]

    @staticmethod
    def set_to_odd(number):
        return number if number % 2 == 1 else number + 1

    def update_maze(self):
        # Redraw the maze to update it
        self.draw_maze()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = MazeGenerator()
    app.run()
