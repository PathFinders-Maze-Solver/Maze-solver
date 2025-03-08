import time
from utils import get_neighbors, reconstruct_path

def solve_maze_dfs(start,goal, draw_path,execution_time_label,grid,rows,cols,index):
    start_time = time.time()
    stack = [(start, [start])]  # Stack stores tuples of (current cell, current path)
    visited = set()  # Set to keep track of visited cells
    visited.add(start)

    while stack:
        current, path = stack.pop()  # Get the last cell and its path from the stack
        if current == goal:
            # Draw the final path and update execution time
            draw_path(path)
            return  # Stop once the goal is found

        # Explore all neighbors
        for neighbor in get_neighbors(current,grid,rows,cols,index):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append((neighbor, path + [neighbor]))  # Add the neighbor and the updated path to the stack

    
    execution_time_label.config(text=f"Execution Time: {time.time() - start_time:.4f}s")
