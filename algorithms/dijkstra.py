import heapq

def find_path_Dijkstras(maze):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    start = (1, 0)
    end = (maze.shape[0]-2, maze.shape[1]-1)
    
    pq = []
    heapq.heappush(pq, (0, start))
    distances = {start: 0}
    previous = {start: None}

    while pq:
        current_distance, current_cell = heapq.heappop(pq)

        if current_cell == end:
            path = []
            while current_cell:
                path.append(current_cell)
                current_cell = previous[current_cell]
            return path[::-1]

        for dx, dy in directions:
            next_cell = (current_cell[0] + dx, current_cell[1] + dy)

            if (0 <= next_cell[0] < maze.shape[0] and 
                0 <= next_cell[1] < maze.shape[1] and 
                maze[next_cell] == 0):
                
                new_distance = current_distance + 1
                if next_cell not in distances or new_distance < distances[next_cell]:
                    distances[next_cell] = new_distance
                    previous[next_cell] = current_cell
                    heapq.heappush(pq, (new_distance, next_cell))

    return None
