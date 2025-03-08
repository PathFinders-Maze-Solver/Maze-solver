import time
from collections import deque
from utils import get_neighbors, reconstruct_path

def solve_maze_bfs(start,goal, draw_path,execution_time_label,grid,rows,cols,index): 
    start_time = time.time()
    queue = deque([start])
    came_from = {}
    visited = set()
    visited.add(start)
    
    while queue:
        current = queue.popleft()
        if current == goal:
            path = reconstruct_path(came_from, goal)
            draw_path(path)
            execution_time_label.config(text=f"Execution Time: {time.time() - start_time:.4f}s")
            return
        
        for neighbor in get_neighbors(current,grid,rows,cols,index):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

