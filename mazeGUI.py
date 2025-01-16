import tkinter as tk
import random
import heapq
from collections import deque

class MazeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Maze Generator & Pathfinding")
        self.geometry("600x600")
        
        # Create a frame to hold the canvas and buttons
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(pady=10, expand=True, fill=tk.BOTH)

        # Canvas for maze
        self.canvas = tk.Canvas(self.main_frame, bg="white")
        self.canvas.pack(side=tk.TOP, padx=10, pady=10, expand=True, fill=tk.BOTH)

        # Size input for the maze
        self.size_label = tk.Label(self.main_frame, text="Maze Size (Odd numbers only):")
        self.size_label.pack()

        self.size_entry = tk.Entry(self.main_frame)
        self.size_entry.pack(pady=10)

        self.generate_button = tk.Button(self.main_frame, text="Generate Maze", command=self.generate_maze)
        self.generate_button.pack(pady=5)

        # Pathfinding algorithm selection
        self.algorithm_label = tk.Label(self.main_frame, text="Choose Pathfinding Algorithm:")
        self.algorithm_label.pack()

        self.algorithm_var = tk.StringVar(value="BFS")
        self.bfs_radio = tk.Radiobutton(self.main_frame, text="BFS", variable=self.algorithm_var, value="BFS")
        self.a_star_radio = tk.Radiobutton(self.main_frame, text="A*", variable=self.algorithm_var, value="A*")
        self.dijkstra_radio = tk.Radiobutton(self.main_frame, text="Dijkstra", variable=self.algorithm_var, value="Dijkstra")
        
        self.bfs_radio.pack()
        self.a_star_radio.pack()
        self.dijkstra_radio.pack()

        self.solve_button = tk.Button(self.main_frame, text="Solve Maze", command=self.solve_maze)
        self.solve_button.pack(pady=5)

        self.reset_button = tk.Button(self.main_frame, text="Reset Maze", command=self.reset_maze)
        self.reset_button.pack(pady=5)

        # Initializing maze variables
        self.maze = None
        self.cell_size = 20
        self.width = 19
        self.height = 19
        self.start = None
        self.end = None
        self.path = []

    def generate_maze(self):
        self.width = int(self.size_entry.get()) if self.size_entry.get() else 19
        self.height = self.width
        if self.width % 2 == 0:
            self.width += 1
        if self.height % 2 == 0:
            self.height += 1

        self.maze = [[1 for _ in range(self.width)] for _ in range(self.height)]
        self.start = (1, 1)
        self.end = (self.width - 2, self.height - 2)
        
        self.carve_passages(self.start[0], self.start[1])

        self.update_canvas()
        self.animate_backtracking(self.start[0], self.start[1])

    def carve_passages(self, cx, cy):
        directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]
        random.shuffle(directions)

        for direction in directions:
            nx, ny = cx + direction[0], cy + direction[1]
            if 0 <= nx < self.width and 0 <= ny < self.height and self.maze[ny][nx] == 1:
                self.maze[cy][cx] = 0
                self.maze[ny][nx] = 0
                self.carve_passages(nx, ny)

    def animate_backtracking(self, cx, cy):
        if (cx, cy) == self.end:
            return
        directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]
        random.shuffle(directions)

        for direction in directions:
            nx, ny = cx + direction[0], cy + direction[1]
            if 0 <= nx < self.width and 0 <= ny < self.height and self.maze[ny][nx] == 0:
                self.maze[ny][nx] = 2
                self.update_canvas()
                self.after(50, self.animate_backtracking, nx, ny)
                return

    def update_canvas(self):
        if self.canvas.winfo_exists():
            self.canvas.delete("all")
            for y in range(self.height):
                for x in range(self.width):
                    color = "black" if self.maze[y][x] == 1 else "white"
                    if self.maze[y][x] == 2:
                        color = "gray"
                    self.canvas.create_rectangle(x * self.cell_size, y * self.cell_size,
                                                 (x + 1) * self.cell_size, (y + 1) * self.cell_size,
                                                 fill=color, outline="gray")
            self.canvas.create_rectangle(self.start[0] * self.cell_size, self.start[1] * self.cell_size,
                                         (self.start[0] + 1) * self.cell_size, (self.start[1] + 1) * self.cell_size,
                                         fill="green")
            self.canvas.create_rectangle(self.end[0] * self.cell_size, self.end[1] * self.cell_size,
                                         (self.end[0] + 1) * self.cell_size, (self.end[1] + 1) * self.cell_size,
                                         fill="red")

    def solve_maze(self):
        if self.algorithm_var.get() == "BFS":
            self.path = self.bfs()
        elif self.algorithm_var.get() == "A*":
            self.path = self.a_star()
        elif self.algorithm_var.get() == "Dijkstra":
            self.path = self.dijkstra()
        
        self.animate_solution()

    def bfs(self):
        queue = deque([self.start])
        came_from = {self.start: None}
        while queue:
            current = queue.popleft()
            if current == self.end:
                break
            for neighbor in self.get_neighbors(current):
                if neighbor not in came_from:
                    queue.append(neighbor)
                    came_from[neighbor] = current

        path = []
        current = self.end
        while current != self.start:
            path.append(current)
            current = came_from[current]
        path.append(self.start)
        path.reverse()
        return path

    def a_star(self):
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_list = []
        heapq.heappush(open_list, (0 + heuristic(self.start, self.end), 0, self.start))
        g_costs = {self.start: 0}
        came_from = {self.start: None}
        
        while open_list:
            _, g_cost, current = heapq.heappop(open_list)
            if current == self.end:
                break
            for neighbor in self.get_neighbors(current):
                tentative_g_cost = g_cost + 1
                if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                    g_costs[neighbor] = tentative_g_cost
                    f_cost = tentative_g_cost + heuristic(neighbor, self.end)
                    heapq.heappush(open_list, (f_cost, tentative_g_cost, neighbor))
                    came_from[neighbor] = current

        path = []
        current = self.end
        while current != self.start:
            path.append(current)
            current = came_from[current]
        path.append(self.start)
        path.reverse()
        return path

    def dijkstra(self):
        open_list = []
        heapq.heappush(open_list, (0, self.start))
        g_costs = {self.start: 0}
        came_from = {self.start: None}

        while open_list:
            g_cost, current = heapq.heappop(open_list)
            if current == self.end:
                break
            for neighbor in self.get_neighbors(current):
                tentative_g_cost = g_cost + 1
                if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                    g_costs[neighbor] = tentative_g_cost
                    heapq.heappush(open_list, (tentative_g_cost, neighbor))
                    came_from[neighbor] = current

        path = []
        current = self.end
        while current != self.start:
            path.append(current)
            current = came_from[current]
        path.append(self.start)
        path.reverse()
        return path

    def get_neighbors(self, node):
        x, y = node
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height and self.maze[ny][nx] != 1:
                neighbors.append((nx, ny))
        return neighbors

    def animate_solution(self):
        for (x, y) in self.path:
            self.maze[y][x] = 3
            self.update_canvas()
            self.after(100)

    def reset_maze(self):
        self.maze = None
        self.update_canvas()

if __name__ == "__main__":
    app = MazeApp()
    app.mainloop()
