# utils.py

def get_neighbors(cell,grid,rows,cols,index):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors = []
    for dx, dy in directions:
        ni, nj = cell.i + dx, cell.j + dy
        if 0 <= ni < cols and 0 <= nj < rows:
            neighbor = grid[index(ni, nj)]
            if not neighbor.is_wall:
                neighbors.append(neighbor)
    return neighbors

def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(current)  # Add the start point
    path.reverse()  # Reverse to get the path from start to goal
    return path
