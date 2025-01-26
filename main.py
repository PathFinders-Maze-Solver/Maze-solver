import tkinter as tk
from tkinter import messagebox
from queue import Queue
import heapq
import random
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from algorithms.bfs import find_path_BFS
from algorithms.dijkstra import find_path_Dijkstras
from algorithms.a_star import find_path_A_star


def create_maze(dim):
    # Create a grid filled with walls
    maze = np.ones((dim*2+1, dim*2+1))

    # Define the starting point
    x, y = (0, 0)
    maze[2*x+1, 2*y+1] = 0

    # Initialize the stack with the starting point
    stack = [(x, y)]
    while len(stack) > 0:
        x, y = stack[-1]

        # Define possible directions
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if nx >= 0 and ny >= 0 and nx < dim and ny < dim and maze[2*nx+1, 2*ny+1] == 1:
                maze[2*nx+1, 2*ny+1] = 0
                maze[2*x+1+dx, 2*y+1+dy] = 0
                stack.append((nx, ny))
                break
        else:
            stack.pop()
            
    # Create an entrance and an exit
    maze[1, 0] = 0
    maze[-2, -1] = 0

    return maze


def draw_maze(maze, path=None):
    fig, ax = plt.subplots(figsize=(10,10))
    
    # Set the border color to white
    fig.patch.set_edgecolor('white')
    fig.patch.set_linewidth(0)

    ax.imshow(maze, cmap=plt.cm.binary, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Prepare for path animation
    if path is not None:
        line, = ax.plot([], [], color='red', linewidth=2)
        
        def init():
            line.set_data([], [])
            return line,
        
        # update is called for each path point in the maze
        def update(frame):
            x, y = path[frame]
            line.set_data(*zip(*[(p[1], p[0]) for p in path[:frame+1]]))  # update the data
            return line,
        
        ani = animation.FuncAnimation(fig, update, frames=range(len(path)), init_func=init, blit=True, repeat = False, interval=20)
    
    # Draw entry and exit arrows
    ax.arrow(0, 1, .4, 0, fc='green', ec='green', head_width=0.3, head_length=0.3)
    ax.arrow(maze.shape[1] - 1, maze.shape[0]  - 2, 0.4, 0, fc='blue', ec='blue', head_width=0.3, head_length=0.3)
    plt.show()

# Global variables to store maze and path
current_maze = None
current_path = None
canvas = None

# Function to handle the button click event for generating maze
def on_generate_maze():
    global current_maze, current_path, canvas
    try:
        dim = int(dim_entry.get())
        algo = algo_var.get()

        if dim <= 0:
            raise ValueError("Maze dimension must be positive")

        current_maze = create_maze(dim)

        if algo == 1:
            current_path = find_path_BFS(current_maze)
        elif algo == 2:
            current_path = find_path_Dijkstras(current_maze)
        elif algo == 3:
            current_path = find_path_A_star(current_maze)
        else:
            messagebox.showerror("Invalid Algorithm", "Please select a valid algorithm.")
            return

        if current_path is not None:
            fig = draw_maze(current_maze, current_path)
            canvas = FigureCanvasTkAgg(fig, master=window)
            canvas.draw()
            canvas.get_tk_widget().pack(padx=10, pady=10)
        else:
            messagebox.showerror("No Path", "No path found!")
    except ValueError as e:
        messagebox.showerror("Invalid Input", str(e))

# Function to reset the visualization
def reset_visualization():
    global current_maze, current_path, canvas
    if current_maze is not None:
        current_path = None  # Clear the path
        canvas.get_tk_widget().destroy()  # Remove the old maze
        fig = draw_maze(current_maze)  # Draw the maze without a path
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(padx=10, pady=10)

# Function to clear the maze and path
def clear_maze():
    global current_maze, current_path, canvas
    current_maze = None
    current_path = None
    canvas.get_tk_widget().destroy()

# Function to remove the path
def remove_path():
    global current_path, canvas
    if current_path is not None:
        current_path = None
        canvas.get_tk_widget().destroy()
        fig = draw_maze(current_maze)  # Redraw maze without path
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(padx=10, pady=10)

# Create the GUI window
window = tk.Tk()
window.title("Maze Generator and Pathfinding")
window.geometry("600x600")

# Create the UI components
dim_label = tk.Label(window, text="Enter Maze Dimension (n):")
dim_label.pack(pady=10)

dim_entry = tk.Entry(window)
dim_entry.pack(pady=5)

algo_label = tk.Label(window, text="Select Pathfinding Algorithm:")
algo_label.pack(pady=10)

algo_var = tk.IntVar()
algo_var.set(1)  # Default to BFS
algo_bfs = tk.Radiobutton(window, text="BFS (Breadth-First Search)", variable=algo_var, value=1)
algo_bfs.pack()
algo_dijkstra = tk.Radiobutton(window, text="Dijkstra's Algorithm", variable=algo_var, value=2)
algo_dijkstra.pack()
algo_astar = tk.Radiobutton(window, text="A* (A-star)", variable=algo_var, value=3)
algo_astar.pack()

generate_button = tk.Button(window, text="Generate Maze and Find Path", command=on_generate_maze)
generate_button.pack(pady=20)

clear_button = tk.Button(window, text="Clear Maze", command=clear_maze)
clear_button.pack(pady=5)

remove_button = tk.Button(window, text="Remove Path", command=remove_path)
remove_button.pack(pady=5)

# Start the GUI event loop
window.mainloop()
