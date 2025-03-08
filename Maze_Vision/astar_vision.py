import heapq
import time
from utils import get_neighbors, reconstruct_path

def solve_maze_astar(start,goal, draw_path,execution_time_label,grid,rows,cols,index):
    """Solve the maze using A* algorithm."""
    start_time = time.time()
    open_set = [(0, start)]  # Priority queue of (f_score, cell)
    came_from = {}
    g_score = {cell: float('inf') for cell in grid}
    g_score[start] = 0
    f_score = {cell: float('inf') for cell in grid}
    f_score[start] = abs(start.i - goal.i) + abs(start.j - goal.j)

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = reconstruct_path(came_from, goal)
            draw_path(path)
            return

        for neighbor in get_neighbors(current,grid,rows,cols,index):
            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + abs(neighbor.i - goal.i) + abs(neighbor.j - goal.j)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    
    execution_time_label.config(text=f"Execution Time: {time.time() - start_time:.4f}s")


