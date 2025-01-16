import tkinter as tk
import random
from collections import deque
import heapq

class MazeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Maze Generator & Pathfinding")
        self.geometry("600x600")
        
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(pady=10, expand=True, fill=tk.BOTH)

        self.canvas = tk.Canvas(self.main_frame, bg="white")
        self.canvas.pack(side=tk.TOP, padx=10, pady=10, expand=True, fill=tk.BOTH)

        self.size_label = tk.Label(self.main_frame, text="Maze Size (Odd numbers only):")
        self.size_label.pack()

        self.size_entry = tk.Entry(self.main_frame)
        self.size_entry.pack(pady=10)

        self.generate_button = tk.Button(self.main_frame, text="Generate Maze", command=self.start_maze_generation)
        self.generate_button.pack(pady=5)

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

        self.maze = None
        self.cell_size = 20
        self.width = 19
        self.height = 19
        self.start = None
        self.end = None
        self.path = []
        self.generation_queue = []

    def start_maze_generation(self):
        self.width = int(self.size_entry.get()) if self.size_entry.get() else 19
        self.height = self.width
        if self.width % 2 == 0:
            self.width += 1
        if self.height % 2 == 0:
            self.height += 1

        self.maze = [[1 for _ in range(self.width)] for _ in range(self.height)]
        self.start = (1, 1)
        self.end = (self.width - 2, self.height - 2)
        self.generation_queue.append(self.start)
        self.generate_step()

    def generate_step(self):
        if not self.generation_queue:
            return

        cx, cy = self.generation_queue.pop(0)
        self.maze[cy][cx] = 0
        directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]
        random.shuffle(directions)

        for direction in directions:
            nx, ny = cx + direction[0], cy + direction[1]
            if 0 <= nx < self.width and 0 <= ny < self.height and self.maze[ny][nx] == 1:
                mx, my = cx + direction[0] // 2, cy + direction[1] // 2
                self.maze[my][mx] = 0
                self.maze[ny][nx] = 0
                self.generation_queue.append((nx, ny))

        self.update_canvas()
        self.after(50, self.generate_step)

    def update_canvas(self):
        if self.canvas.winfo_exists():
            self.canvas.delete("all")
            for y in range(self.height):
                for x in range(self.width):
                    color = "black" if self.maze[y][x] == 1 else "white"
                    if self.maze[y][x] == 2:
                        color = "gray"
                    if self.maze[y][x] == 3:
                        color = "blue"
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
            self.maze[current[1]][current[0]] = 2  # Mark as visited
            self.update_canvas()
            self.update_idletasks()
            self.after(10)

            if current == self.end:
                break
            for neighbor in self.get_neighbors(current):
                if neighbor not in came_from:
                    queue.append(neighbor)
                    came_from[neighbor] = current

        return self.reconstruct_path(came_from)

    def a_star(self):
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_list = []
        heapq.heappush(open_list, (0 + heuristic(self.start, self.end), 0, self.start))
        g_costs = {self.start: 0}
        came_from = {self.start: None}
        
        while open_list:
            _, g_cost, current = heapq.heappop(open_list)
            self.maze[current[1]][current[0]] = 2  # Mark as visited
            self.update_canvas()
            self.update_idletasks()
            self.after(10)

            if current == self.end:
                break
            for neighbor in self.get_neighbors(current):
                tentative_g_cost = g_cost + 1
                if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                    g_costs[neighbor] = tentative_g_cost
                    f_cost = tentative_g_cost + heuristic(neighbor, self.end)
                    heapq.heappush(open_list, (f_cost, tentative_g_cost, neighbor))
                    came_from[neighbor] = current

        return self.reconstruct_path(came_from)

    def dijkstra(self):
        open_list = []
        heapq.heappush(open_list, (0, self.start))
        g_costs = {self.start: 0}
        came_from = {self.start: None}

        while open_list:
            g_cost, current = heapq.heappop(open_list)
            self.maze[current[1]][current[0]] = 2  # Mark as visited
            self.update_canvas()
            self.update_idletasks()
            self.after(10)

            if current == self.end:
                break
            for neighbor in self.get_neighbors(current):
                tentative_g_cost = g_cost + 1
                if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                    g_costs[neighbor] = tentative_g_cost
                    heapq.heappush(open_list, (tentative_g_cost, neighbor))
                    came_from[neighbor] = current

        return self.reconstruct_path(came_from)

    def get_neighbors(self, node):
        x, y = node
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height and self.maze[ny][nx] == 0:
                neighbors.append((nx, ny))
        return neighbors

    def reconstruct_path(self, came_from):
        path = []
        current = self.end
        while current != self.start:
            path.append(current)
            current = came_from[current]
        path.append(self.start)
        path.reverse()
        return path

    def animate_solution(self):
        for (x, y) in self.path:
            self.maze[y][x] = 3
            self.update_canvas()
            self.update_idletasks()
            self.after(100)

    def reset_maze(self):
        self.maze = None
        self.update_canvas()

if __name__ == "__main__":
    app = MazeApp()
    app.mainloop()