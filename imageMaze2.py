# import matplotlib.animation as animation
# import matplotlib.pyplot as plt
# import numpy as np
# from queue import Queue
# from PIL import Image

# def load_maze_image(image_path):
#     # Open the maze image and convert it to greyscale
#     im = Image.open(image_path).convert('L')
#     # Ensure all black pixels are 0 and all white pixels are 1
#     binary = im.point(lambda p: p > 128 and 1)
#     # Convert to Numpy array
#     maze = np.array(binary)
#     return maze

# def find_start_and_end(maze):
#     # Find the start (first open path) and end (last open path)
#     start = None
#     end = None
    
#     # Locate start point
#     for y in range(maze.shape[0]):
#         for x in range(maze.shape[1]):
#             if maze[y, x] == 0:  # Open path
#                 start = (y, x)
#                 break
#         if start:
#             break

#     # Locate end point
#     for y in range(maze.shape[0]-1, -1, -1):
#         for x in range(maze.shape[1]-1, -1, -1):
#             if maze[y, x] == 0:  # Open path
#                 end = (y, x)
#                 break
#         if end:
#             break

#     return start, end

# def find_path(maze, start, end):
#     # BFS algorithm to find the shortest path
#     directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
#     visited = np.zeros_like(maze, dtype=bool)
#     visited[start] = True
#     queue = Queue()
#     queue.put((start, []))
    
#     while not queue.empty():
#         (node, path) = queue.get()
#         for dx, dy in directions:
#             next_node = (node[0]+dx, node[1]+dy)
#             if (next_node == end):
#                 return path + [next_node]
#             if (next_node[0] >= 0 and next_node[1] >= 0 and 
#                 next_node[0] < maze.shape[0] and next_node[1] < maze.shape[1] and 
#                 maze[next_node] == 0 and not visited[next_node]):
#                 visited[next_node] = True
#                 queue.put((next_node, path + [next_node]))

#     return None  # Return None if no path is found

# def draw_maze(maze, path=None, start=None, end=None):
#     fig, ax = plt.subplots(figsize=(10, 10))
    
#     # Set the border color to black
#     fig.patch.set_edgecolor('black')
#     fig.patch.set_linewidth(0)

#     # Display the maze: walls as black (1) and paths as white (0)
#     ax.imshow(maze, cmap='gray', interpolation='nearest')  # Use gray colormap
#     ax.set_xticks([])
#     ax.set_yticks([])
    
#     # Draw the path if it exists
#     if path is not None:
#         for (y, x) in path:
#             ax.plot(x, y, marker='o', color='white', markersize=5)  # Draw each point of the path

#         # Prepare for path animation
#         line, = ax.plot([], [], color='white', linewidth=2)
        
#         def init():
#             line.set_data([], [])
#             return line,
        
#         # Update is called for each path point in the maze
#         def update(frame):
#             x, y = path[frame]
#             line.set_data(*zip(*[(p[1], p[0]) for p in path[:frame+1]]))  # update the data
#             return line,

#         ani = animation.FuncAnimation(fig, update, frames=range(len(path)), init_func=init, blit=True, repeat=False, interval=200)
    
#     # Draw entry (start) and exit (end) markers
#     if start:
#         ax.plot(start[1], start[0], marker='o', color='green', markersize=10, label='Start')  # Start point
#     if end:
#         ax.plot(end[1], end[0], marker='o', color='blue', markersize=10, label='End')  # End point
    
#     # Add a legend to indicate start and end
#     ax.legend(loc='upper right')

#     plt.show()

# if __name__ == "__main__":
#     maze = load_maze_image('maze10.png')
#     start, end = find_start_and_end(maze)
#     path = find_path(maze, start, end)
#     draw_maze(maze, path, start, end)


import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from queue import Queue
from PIL import Image

def load_maze_image(image_path):
    # Open the maze image and convert it to greyscale
    im = Image.open(image_path).convert('L')
    
    # Ensure all black pixels are 0 and all white pixels are 1
    binary = im.point(lambda p: p > 128 and 1)
    
    # Convert to Numpy array
    maze = np.array(binary)

    # Add a border of walls (1s) around the maze
    maze_with_border = np.ones((maze.shape[0] + 2, maze.shape[1] + 2), dtype=int)
    maze_with_border[1:-1, 1:-1] = maze  # Place the original maze in the center
    return maze_with_border

def find_start_and_end(maze):
    # Find the start (first open path) and end (last open path)
    start = None
    end = None
    
    # Locate start point
    for y in range(maze.shape[0]):
        for x in range(maze.shape[1]):
            if maze[y, x] == 0:  # Open path
                start = (y, x)
                break
        if start:
            break

    # Locate end point
    for y in range(maze.shape[0]-1, -1, -1):
        for x in range(maze.shape[1]-1, -1, -1):
            if maze[y, x] == 0:  # Open path
                end = (y, x)
                break
        if end:
            break

    return start, end

def find_path(maze, start, end):
    # BFS algorithm to find the shortest path
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    visited = np.zeros_like(maze, dtype=bool)
    visited[start] = True
    queue = Queue()
    queue.put((start, []))
    
    while not queue.empty():
        (node, path) = queue.get()
        for dx, dy in directions:
            next_node = (node[0]+dx, node[1]+dy)
            if (next_node == end):
                return path + [next_node]
            if (next_node[0] >= 0 and next_node[1] >= 0 and 
                next_node[0] < maze.shape[0] and next_node[1] < maze.shape[1] and 
                maze[next_node] == 0 and not visited[next_node]):
                visited[next_node] = True
                queue.put((next_node, path + [next_node]))

    return None  # Return None if no path is found

# def draw_maze(maze, path=None, start=None, end=None):
#     fig, ax = plt.subplots(figsize=(10, 10))
    
#     # Set the border color to black
#     fig.patch.set_edgecolor('black')
#     fig.patch.set_linewidth(0)

#     # Display the maze: walls as black (1) and paths as white (0)
#     ax.imshow(maze, cmap='gray', interpolation='nearest')  # Use gray colormap
#     ax.set_xticks([])
#     ax.set_yticks([])
    
#     # Draw the path if it exists
#     if path is not None:
#         for (y, x) in path:
#             ax.plot(x, y, marker='o', color='white', markersize=5)  # Draw each point of the path

#         # Prepare for path animation
#         line, = ax.plot([], [], color='white', linewidth=2)
        
#         def init():
#             line.set_data([], [])
#             return line,
        
#         # Update is called for each path point in the maze
#         def update(frame):
#             x, y = path[frame]
#             line.set_data(*zip(*[(p[1], p[0]) for p in path[:frame+1]]))  # update the data
#             return line,

#         ani = animation.FuncAnimation(fig, update, frames=range(len(path)), init_func=init, blit=True, repeat=False, interval=200)
    
#     # Draw entry (start) and exit (end) markers
#     if start:
#         ax.plot(start[1], start[0], marker='o', color='green', markersize=10, label='Start')  # Start point
#     if end:
#         ax.plot(end[1], end[0], marker='o', color='blue', markersize=10, label='End')  # End point
    
#     # Add a legend to indicate start and end
#     ax.legend(loc='upper right')

#     plt.show()

# Re-defining the draw_maze function since it wasn't in this notebook execution context

def draw_maze(maze, path=None, start=None, end=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set the border color to black
    fig.patch.set_edgecolor('black')
    fig.patch.set_linewidth(0)

    # Display the maze: walls as black (1) and paths as white (0)
    ax.imshow(maze, cmap='gray', interpolation='nearest')  # Use gray colormap
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Draw the path if it exists
    if path is not None:
        for (y, x) in path:
            ax.plot(x, y, marker='o', color='white', markersize=5)  # Draw each point of the path

        # Prepare for path animation
        line, = ax.plot([], [], color='white', linewidth=2)
        
        def init():
            line.set_data([], [])
            return line,
        
        # Update is called for each path point in the maze
        def update(frame):
            x, y = path[frame]
            line.set_data(*zip(*[(p[1], p[0]) for p in path[:frame+1]]))  # update the data
            return line,

        ani = animation.FuncAnimation(fig, update, frames=range(len(path)), init_func=init, blit=True, repeat=False, interval=200)
    
    # Draw entry (start) and exit (end) markers
    if start:
        ax.plot(start[1], start[0], marker='o', color='green', markersize=10, label='Start')  # Start point
    if end:
        ax.plot(end[1], end[0], marker='o', color='blue', markersize=10, label='End')  # End point
    
    # Add a legend to indicate start and end
    ax.legend(loc='upper right')

    plt.show()


if __name__ == "__main__":
    maze = load_maze_image('maze.png')
    start, end = find_start_and_end(maze)
    
    # Adjust the start and end for the border added
    adjusted_start = (start[0] + 1, start[1] + 1)
    adjusted_end = (end[0] + 1, end[1] + 1)

    path = find_path(maze, adjusted_start, adjusted_end)
    draw_maze(maze, path, adjusted_start, adjusted_end)
