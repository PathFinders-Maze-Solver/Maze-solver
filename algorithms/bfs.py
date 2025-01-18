from queue import Queue

import numpy as np

# BFS Algorithm
def find_path_BFS(maze):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    start = (1, 1)
    end = (maze.shape[0]-2, maze.shape[1]-2)
    visited = np.zeros_like(maze, dtype=bool)
    visited[start] = True
    queue = Queue()
    queue.put((start, []))
    while not queue.empty():
        (node, path) = queue.get()
        for dx, dy in directions:
            next_node = (node[0] + dx, node[1] + dy)
            if (next_node == end):
                return path + [next_node]
            if (next_node[0] >= 0 and next_node[1] >= 0 and 
                next_node[0] < maze.shape[0] and next_node[1] < maze.shape[1] and 
                maze[next_node] == 0 and not visited[next_node]):
                visited[next_node] = True
                queue.put((next_node, path + [next_node]))

