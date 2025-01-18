import heapq

def find_path_A_star(maze):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    start = (1, 1)
    end = (maze.shape[0]-2, maze.shape[1]-2)
    
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_list = []
    heapq.heappush(open_list, (0 + heuristic(start, end), 0, start))  # f, g, node
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}

    while open_list:
        _, current_g, current = heapq.heappop(open_list)

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            
            if (0 <= neighbor[0] < maze.shape[0] and 
                0 <= neighbor[1] < maze.shape[1] and 
                maze[neighbor] == 0):

                tentative_g_score = current_g + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                    heapq.heappush(open_list, (f_score[neighbor], tentative_g_score, neighbor))
                    came_from[neighbor] = current

    return None
