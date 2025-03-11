import time
from timeit import Timer
import pygame
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
from typing import TYPE_CHECKING, Tuple, List, Set, Dict, Optional
import cv2
import numpy as np
import logging
import os
from collections import Counter
from PIL import ImageOps

# Import your maze solving algorithms
from a_star import solve_maze_a_star
from bfs import solve_maze_bfs
from dfs import solve_maze_dfs
from dijkstra import solve_maze_dijkstra

# Global variables for maze state
cols, rows = 0, 0
grid = []  # List of Cell objects
stack = []  # For generation
current = None  # For generation
start = None  # Maze entry cell
goal = None  # Maze exit cell
grid_size = None # Grid size
w = 0  # Cell size (will be computed)
x_offset = 0  # Maze drawing x offset
y_offset = 0  # Maze drawing y offset
width = 600  # Canvas width
height = 600  # Canvas height


start_time = None  # Used for timing maze generation


# Add a setup function to initialize the content in the provided frame
def setup(parent_frame):
    global root, canvas, execution_time_label, generate_button, algorithm_var, solve_button

    # --- Global Cell class used by both generation and image input ---
    class Cell:
        def __init__(self, i, j):
            self.i = i
            self.j = j
            # Walls: [top, right, bottom, left]
            self.walls = [True, True, True, True]
            self.visited = False
            self.parent = None
            self.rank = 0
            self.f_score = float('inf')
            self.g_score = float('inf')

        def __lt__(self, other):
            return self.f_score < other.f_score

        def __eq__(self, other):
            return self.i == other.i and self.j == other.j

        def __hash__(self):
            return hash((self.i, self.j))

        def show(self, surface, is_start=False, is_goal=False):
            global w, x_offset, y_offset
            # print(w,x_offset,y_offset)
            x = self.i * w + x_offset
            y = self.j * w + y_offset
            # Draw cell background
            pygame.draw.rect(surface, (255, 255, 255), (x, y, w, w))
            # Color the start/goal cells
            if is_start:
                pygame.draw.rect(surface, (0, 255, 0), (x, y, w, w))  # Green for start
            elif is_goal:
                pygame.draw.rect(surface, (255, 0, 0), (x, y, w, w))  # Red for goal
            # Draw walls (black lines)
            if self.walls[0]:
                pygame.draw.line(surface, (0, 0, 0), (x, y), (x + w, y), 2)
            if self.walls[1]:
                pygame.draw.line(surface, (0, 0, 0), (x + w, y), (x + w, y + w), 2)
            if self.walls[2]:
                pygame.draw.line(surface, (0, 0, 0), (x + w, y + w), (x, y + w), 2)
            if self.walls[3]:
                pygame.draw.line(surface, (0, 0, 0), (x, y + w), (x, y), 2)

        def highlight(self, surface):
            global w, x_offset, y_offset
            x = self.i * w + x_offset
            y = self.j * w + y_offset
            pygame.draw.rect(surface, (200, 200, 200), (x, y, w, w))

    # --- MazeProcessor class for image input ---
    class MazeProcessor:
        """
        Area-based wall detection
        cell size detection
        """

        def __init__(self, timer: Optional['Timer'] = None):
            """
            Initialize the MazeProcessor.

            Args:
                timer: performance measurement using Timer object
            """
            self.timer = timer
            self.debug_mode = False

            # Processing parameters
            self.wall_threshold = 127  # Threshold for wall detection (0-255)
            self.invert_binary = False  # Whether to invert the binary image
            self.wall_detection_threshold_h = 0.4  # Threshold for horizontal wall detection
            self.wall_detection_threshold_v = 0.4  # Threshold for vertical wall detection

        def process_image(self, image_path: str, grid_size: Optional[int] = None) -> Tuple[List[Cell], int, int]:
            """
            Process a maze image and return a list of Cell objects along with dimensions.

            Args:
                image_path: Path to the maze image file.
                grid_size: Optional integer indicating the number of cells per row/column.
                        If provided, the cell size is computed dynamically.

            Returns:
                Tuple of (List[Cell], rows, cols)
            """
            if self.timer:
                self.timer.start('processing', 'image')

            # Create debug directory if needed
            if self.debug_mode and not os.path.exists("debug"):
                os.makedirs("debug")

            # Step 1: Load and preprocess the image using PIL
            logging.debug(f"Loading image: {image_path}")
            image = Image.open(image_path).convert("L")
            binary_image = ImageOps.autocontrast(image).point(
                lambda pixel: 0 if pixel < image.getextrema()[0] + 50 else 1, mode="1")
            width, height = binary_image.size
            pixel_matrix = binary_image.load()

            if self.debug_mode:
                # Convert PIL image to OpenCV format for debug saving
                binary_np = np.array(binary_image, dtype=np.uint8) * 255
                cv2.imwrite("debug/01_original.png", binary_np)

            # Step 2: Find start and end points
            start = self._point_start(pixel_matrix, width, height)
            end = self._point_end(pixel_matrix, width, height)

            if self.debug_mode:
                binary_debug = np.array(binary_image, dtype=np.uint8) * 255
                debug_img = cv2.cvtColor(binary_debug, cv2.COLOR_GRAY2BGR)
                if start != (-1, -1):
                    cv2.circle(debug_img, (start[0], start[1]), 5, (0, 0, 255), -1)
                if end != (-1, -1):
                    cv2.circle(debug_img, (end[0], end[1]), 5, (0, 255, 0), -1)
                cv2.imwrite("debug/02_points.png", debug_img)

            # Step 3: Extract maze area
            maze_binary = self._get_maze(pixel_matrix, binary_image, start, end)

            if self.debug_mode:
                maze_debug = np.array(maze_binary, dtype=np.uint8) * 255
                cv2.imwrite("debug/03_maze_area.png", maze_debug)

            # Step 4: Detect grid parameters
            if grid_size is not None and grid_size > 0:
                # Compute cell size dynamically based on grid_size
                cell_size = min(maze_binary.shape[0] // grid_size, maze_binary.shape[1] // grid_size)
                if cell_size <= 0:
                    cell_size = 20  # Fallback to default cell size
                rows, cols = grid_size, grid_size
            else:
                # Use automatic detection of cell size
                path_width, wall_width = self._find_min_spacing(maze_binary)
                cell_size = path_width + wall_width
                rows, cols, offset_y, offset_x = self._calculate_grid_dimensions(maze_binary, path_width, wall_width)

            logging.debug(f"Detected grid: cell_size={cell_size}, dimensions={rows}x{cols}")

            if self.debug_mode:
                maze_debug = np.array(maze_binary, dtype=np.uint8) * 255
                grid_viz = cv2.cvtColor(maze_debug, cv2.COLOR_GRAY2BGR)

                # Draw detected grid
                for r in range(rows + 1):
                    y = r * cell_size
                    if y < grid_viz.shape[0]:
                        cv2.line(grid_viz, (0, y), (grid_viz.shape[1], y), (0, 255, 0), 1)

                for c in range(cols + 1):
                    x = c * cell_size
                    if x < grid_viz.shape[1]:
                        cv2.line(grid_viz, (x, 0), (x, grid_viz.shape[0]), (0, 255, 0), 1)

                cv2.imwrite("debug/04_grid_detection.png", grid_viz)

            # Step 5: Detect walls using area-based approach
            wall_indicators = self._detect_walls_cell_based(maze_binary, rows, cols, cell_size)

            # Step 6: Create a list of Cell objects
            cells = []
            for r in range(rows):
                for c in range(cols):
                    cell = Cell(c, r)
                    cell.walls = wall_indicators[r][c]
                    cells.append(cell)

            if self.debug_mode:
                self._visualize_walls(wall_indicators, rows, cols)

            if self.timer:
                self.timer.stop()

            logging.debug(f"Maze processed successfully: {rows}x{cols} cells")

            return cells, rows, cols

        def _point_start(self, pixel_matrix, width, height):
            """
            Find the starting point of the maze. This function looks for walls that
            extend a significant distance to identify possible entrances.

            Args:
                pixel_matrix: Binary image pixel matrix
                width: Image width
                height: Image height

            Returns:
                Tuple of (x, y) coordinates of the start point, return (-1, -1) if not found
            """
            start = (-1, -1)
            for i in range(width):
                for j in range(height):
                    if pixel_matrix[i, j] == 0:  # Wall pixel found
                        k = j
                        while (k < height and pixel_matrix[i, k] == 0):
                            k += 1
                        l = i
                        while (l < height and pixel_matrix[l, j] == 0):
                            l += 1
                        if k - j > height * 0.5 or l - i > width * 0.5:
                            start = (i, j)
                            break
                if start != (-1, -1):
                    break
            return start

        def _point_end(self, pixel_matrix, width, height):
            """
            Find the ending point of the maze, searching from the bottom-right.

            Args:
                pixel_matrix: Binary image pixel matrix
                width: Image width
                height: Image height

            Returns:
                Tuple of (x, y) coordinates of the end point, or (-1, -1) if not found
            """
            finish = (-1, -1)
            for j in range(height - 1, -1, -1):
                for i in range(width - 1, -1, -1):
                    if pixel_matrix[i, j] == 0:  # Wall pixel found
                        k = j
                        while k >= 0 and pixel_matrix[i, k] == 0:
                            k -= 1
                        l = i
                        while l >= 0 and pixel_matrix[l, j] == 0:
                            l -= 1
                        if (j - k) > height * 0.5 or (i - l) > width * 0.5:
                            finish = (i, j)
                            return finish
            return finish

        def _get_maze(self, pixel_matrix, binary_image, start, end):
            """
            Extract the maze area from the binary image based on start and end points,
            preserving the original orientation.

            Args:
                pixel_matrix: Binary image pixel matrix
                binary_image: Binary image
                start: Start point (x, y)
                end: End point (x, y)

            Returns:
                NumPy array representing the maze area
            """
            width, height = binary_image.size

            # Use full image if start/end not found
            if start == (-1, -1) or end == (-1, -1):
                indexNorth, indexWest = 0, 0
                indexSout, indexEast = width - 1, height - 1
            else:
                indexNorth, indexWest = start
                indexSout, indexEast = end

            # Calculate dimensions and create maze matrix with CORRECT orientation
            # In PIL, coordinates are (x,y) = (width, height) order
            # In NumPy arrays, dimensions are (height, width) order
            h = indexEast - indexWest + 1  # Height (j dimension)
            w = indexSout - indexNorth + 1  # Width (i dimension)

            # Create the maze with proper dimensions (height, width)
            maze = np.zeros((h, w), dtype=np.uint8)

            # Fill maze matrix preserving original orientation
            for i in range(indexNorth, indexSout + 1):
                for j in range(indexWest, indexEast + 1):
                    if 0 <= i < width and 0 <= j < height:
                        # Map to the correct position in the maze array
                        # j-indexWest gives the row (y coordinate in numpy)
                        # i-indexNorth gives the column (x coordinate in numpy)
                        maze[j - indexWest, i - indexNorth] = pixel_matrix[i, j]

            return maze

        def _find_min_spacing(self, maze_binary):
            """
            Find the minimum spacing between walls by analyzing horizontal and vertical distances.

            Args:
                maze_binary: Binary maze image (0=wall, 1=path)

            Returns:
                Tuple of (path_width, wall_width)
            """
            maze_height, maze_width = maze_binary.shape

            # Find horizontal spacing (distances between vertical walls)
            horizontal_spaces = []
            horizontal_walls = []

            for row in range(maze_height):
                # Analyze path spaces
                spaces = []
                current_space = 0
                # Analyze wall segments
                walls = []
                current_wall = 0

                for col in range(maze_width):
                    if maze_binary[row, col] == 1:  # Path pixel
                        # Count path length
                        current_space += 1
                        # End of wall segment check
                        if current_wall > 0:
                            walls.append(current_wall)
                            current_wall = 0
                    else:  # Wall pixel
                        # Count wall length
                        current_wall += 1
                        # End of path segment check
                        if current_space > 0:
                            spaces.append(current_space)
                            current_space = 0

                # Add the last segments if they exist
                if current_space > 0:
                    spaces.append(current_space)
                if current_wall > 0:
                    walls.append(current_wall)

                # Add to overall lists
                horizontal_spaces.extend(spaces)
                horizontal_walls.extend(walls)

            # Find vertical spacing (distances between horizontal walls)
            vertical_spaces = []
            vertical_walls = []

            for col in range(maze_width):
                # Analyze path spaces
                spaces = []
                current_space = 0
                # Analyze wall segments
                walls = []
                current_wall = 0

                for row in range(maze_height):
                    if maze_binary[row, col] == 1:  # Path pixel
                        # Count path length
                        current_space += 1
                        # End of wall segment check
                        if current_wall > 0:
                            walls.append(current_wall)
                            current_wall = 0
                    else:  # Wall pixel
                        # Count wall length
                        current_wall += 1
                        # End of path segment check
                        if current_space > 0:
                            spaces.append(current_space)
                            current_space = 0

                # Add the last segments if they exist
                if current_space > 0:
                    spaces.append(current_space)
                if current_wall > 0:
                    walls.append(current_wall)

                # Add to overall lists
                vertical_spaces.extend(spaces)
                vertical_walls.extend(walls)

            # Filter out very large values (probably outer boundaries)
            max_valid_size = min(maze_height, maze_width) // 4
            horizontal_spaces = [s for s in horizontal_spaces if 0 < s < max_valid_size]
            vertical_spaces = [s for s in vertical_spaces if 0 < s < max_valid_size]
            horizontal_walls = [w for w in horizontal_walls if 0 < w < max_valid_size]
            vertical_walls = [w for w in vertical_walls if 0 < w < max_valid_size]

            # Calculate most common space width (path width)
            if horizontal_spaces and vertical_spaces:
                # Create histograms using Counter to find the most common values
                h_counter = Counter(horizontal_spaces)
                v_counter = Counter(vertical_spaces)

                # Get the most common path width
                h_common = h_counter.most_common(3)
                v_common = v_counter.most_common(3)

                # Select the most reliable (highest count) from small values
                # Sort by count (frequency) and take smallest value with high frequency
                h_candidates = sorted(h_common, key=lambda x: (-x[1], x[0]))
                v_candidates = sorted(v_common, key=lambda x: (-x[1], x[0]))

                path_width_h = h_candidates[0][0] if h_candidates else 1
                path_width_v = v_candidates[0][0] if v_candidates else 1

                # Use the smaller of the two (more likely to be a single path unit)
                path_width = min(path_width_h, path_width_v)
            else:
                path_width = 1  # Default

            # Calculate most common wall width using the same approach
            if horizontal_walls and vertical_walls:
                h_counter = Counter(horizontal_walls)
                v_counter = Counter(vertical_walls)

                h_common = h_counter.most_common(3)
                v_common = v_counter.most_common(3)

                # Favor thinner walls with high frequency
                h_candidates = sorted(h_common, key=lambda x: (-x[1], x[0]))
                v_candidates = sorted(v_common, key=lambda x: (-x[1], x[0]))

                wall_width_h = h_candidates[0][0] if h_candidates else 1
                wall_width_v = v_candidates[0][0] if v_candidates else 1

                # Use the smaller value (more likely to be an actual wall)
                wall_width = min(wall_width_h, wall_width_v)
            else:
                wall_width = 1  # Default

            # Ensure minimum sizes
            path_width = max(1, path_width)
            wall_width = max(1, wall_width)

            if self.debug_mode:
                logging.debug(f"Detected path width: {path_width}, wall width: {wall_width}")
                if 'h_common' in locals() and 'v_common' in locals():
                    logging.debug(f"Horizontal spaces: {h_common}")
                    logging.debug(f"Vertical spaces: {v_common}")
                    logging.debug(f"Horizontal walls: {h_common}")
                    logging.debug(f"Vertical walls: {v_common}")

            return path_width, wall_width

        def _calculate_grid_dimensions(self, maze_matrix, path_width, wall_width):
            """
            Calculate grid dimensions based on cell size with improved edge cell handling.

            Args:
                maze_matrix: Binary maze matrix
                path_width: Width of path
                wall_width: Width of wall

            Returns:
                Tuple of (rows, cols, offset_y, offset_x)
            """
            # In NumPy arrays, shape[0] is height (rows) and shape[1] is width (columns)
            height, width = maze_matrix.shape
            cell_size = path_width + wall_width

            # Calculate number of rows and columns (initial estimate)
            rows_raw = height / cell_size
            cols_raw = width / cell_size

            # Get integer part
            rows_int = int(rows_raw)
            cols_int = int(cols_raw)

            # Calculate the fractional parts (representing partial cells)
            rows_frac = rows_raw - rows_int
            cols_frac = cols_raw - cols_int

            # Determine if we should include the partial row/column
            # Based on threshold (e.g., if more than 50% of cell is present)
            include_partial_row = rows_frac > 0.5
            include_partial_col = cols_frac > 0.5

            # Calculate final dimensions
            rows = rows_int + (1 if include_partial_row else 0)
            cols = cols_int + (1 if include_partial_col else 0)

            # Calculate offsets (to center the grid better)
            offset_y = 0
            offset_x = 0

            # If we're not including a partial row/column but there's some fraction,
            # we can distribute the excess as an offset to center the grid
            if not include_partial_row and rows_frac > 0:
                offset_y = int(rows_frac * cell_size / 2)
            if not include_partial_col and cols_frac > 0:
                offset_x = int(cols_frac * cell_size / 2)

            # Log dimensions for debugging
            if self.debug_mode:
                logging.debug(f"Maze matrix shape: {maze_matrix.shape}")
                logging.debug(f"Cell size: {cell_size}")
                logging.debug(f"Raw dimensions: {rows_raw}x{cols_raw}")
                logging.debug(f"Fractional parts: {rows_frac}x{cols_frac}")
                logging.debug(f"Include partial row/col: {include_partial_row}, {include_partial_col}")
                logging.debug(f"Calculated grid dimensions: {rows}x{cols} with offsets {offset_y},{offset_x}")

            # Ensure we have at least one row and column
            rows = max(1, rows)
            cols = max(1, cols)

            return rows, cols, offset_y, offset_x

        def _detect_walls_cell_based(self, maze_binary, rows, cols, cell_size):
            """
            Detect walls by analyzing entire cells for wall segments and determining
            which edges the walls are closest to, with improved handling for edge cells.

            Args:
                maze_binary: Binary maze image (0=wall, 1=path)
                rows: Number of rows
                cols: Number of columns
                cell_size: Size of each cell

            Returns:
                3D list of wall indicators for each cell [row][col][wall_direction]
            """
            height, width = maze_binary.shape
            wall_indicators = [[[0, 0, 0, 0] for _ in range(cols)] for _ in range(rows)]

            # Compute offsets to better align the grid
            rows_raw = height / cell_size
            cols_raw = width / cell_size
            rows_frac = rows_raw - int(rows_raw)
            cols_frac = cols_raw - int(cols_raw)

            # Calculate offsets to center the grid
            offset_y = int((height - rows * cell_size) / 2) if height > rows * cell_size else 0
            offset_x = int((width - cols * cell_size) / 2) if width > cols * cell_size else 0

            # Debug visualizations
            if self.debug_mode:
                debug_img = np.zeros((height, width, 3), dtype=np.uint8)
                # Make path white
                debug_img[maze_binary == 1] = [255, 255, 255]
                # Make walls gray
                debug_img[maze_binary == 0] = [128, 128, 128]

            # First pass: analyze cells and detect walls
            for r in range(rows):
                for c in range(cols):
                    # Calculate cell boundaries with offset
                    top = int(min(r * cell_size + offset_y, height - 1))
                    left = int(min(c * cell_size + offset_x, width - 1))
                    bottom = int(min((r + 1) * cell_size + offset_y, height))
                    right = int(min((c + 1) * cell_size + offset_x, width))

                    # Skip if out of bounds
                    if top >= height or left >= width:
                        continue

                    # Check if this is a partial cell at the edge of the image
                    is_partial_cell = (right - left < cell_size) or (bottom - top < cell_size)

                    # For partial cells, check if they have enough content to be included
                    if is_partial_cell:
                        # Calculate how much of the cell is actually present
                        cell_area = (right - left) * (bottom - top)
                        full_cell_area = cell_size * cell_size
                        cell_coverage = cell_area / full_cell_area

                        # Extract partial cell
                        partial_cell = maze_binary[top:bottom, left:right]

                        # Count wall and path pixels
                        wall_pixels = np.sum(partial_cell == 0)
                        path_pixels = np.sum(partial_cell == 1)
                        total_pixels = wall_pixels + path_pixels

                        # If the cell has very few path pixels relative to walls, it might be mostly border
                        path_ratio = path_pixels / total_pixels if total_pixels > 0 else 0

                        # Criteria for including a partial cell:
                        # 1. At least 50% of the cell is present AND
                        # 2. There is a reasonable amount of path pixels (>20% of visible area)
                        include_cell = cell_coverage > 0.4 and path_ratio > 0.15

                        if not include_cell:
                            # Mark all walls for this cell and skip detailed analysis
                            wall_indicators[r][c] = [1, 1, 1, 1]  # All walls present
                            continue

                        if self.debug_mode:
                            # Visualize partial cells differently
                            color = (200, 100, 255)  # Purple for partial cells
                            cv2.rectangle(debug_img, (left, top), (right, bottom), color, 1)
                            cv2.putText(debug_img, f"C:{cell_coverage:.2f}", (left + 2, top + 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                            cv2.putText(debug_img, f"P:{path_ratio:.2f}", (left + 2, top + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

                    # Extract cell
                    cell = maze_binary[top:bottom, left:right]

                    # Add cell outline to debug visualization
                    if self.debug_mode:
                        cv2.rectangle(debug_img, (left, top), (right, bottom), (0, 255, 0), 1)

                    # Analyze horizontal walls (look for rows with many wall pixels)
                    h_wall_positions = []
                    for y in range(cell.shape[0]):
                        # Count wall pixels in this row
                        if y < cell.shape[0]:  # Safety check
                            wall_pixels = np.sum(cell[y, :] == 0)

                            # If row is mostly wall pixels (e.g., 70%)
                            if wall_pixels > cell.shape[1] * 0.5:
                                h_wall_positions.append(y)

                    # Group adjacent positions (within a few pixels) to handle thick walls
                    h_walls = []
                    if h_wall_positions:
                        current_wall = [h_wall_positions[0]]
                        for pos in h_wall_positions[1:]:
                            if pos - current_wall[-1] <= 3:  # Adjust threshold as needed
                                current_wall.append(pos)
                            else:
                                h_walls.append(sum(current_wall) / len(current_wall))  # Average position
                                current_wall = [pos]
                        if current_wall:
                            h_walls.append(sum(current_wall) / len(current_wall))  # Add last wall

                    # Assign walls to top or bottom edge based on position
                    for wall_pos in h_walls:
                        if wall_pos < cell.shape[0] / 2:
                            wall_indicators[r][c][0] = 1  # Top wall
                            if self.debug_mode:
                                y_pos = int(top + wall_pos)
                                cv2.line(debug_img, (left, y_pos), (right, y_pos), (0, 0, 255), 1)
                        else:
                            wall_indicators[r][c][2] = 1  # Bottom wall
                            if self.debug_mode:
                                y_pos = int(top + wall_pos)
                                cv2.line(debug_img, (left, y_pos), (right, y_pos), (0, 0, 255), 1)

                    # Analyze vertical walls (look for columns with many wall pixels)
                    v_wall_positions = []
                    for x in range(cell.shape[1]):
                        # Count wall pixels in this column
                        if x < cell.shape[1]:  # Safety check
                            wall_pixels = np.sum(cell[:, x] == 0)

                            # If column is mostly wall pixels
                            if wall_pixels > cell.shape[0] * 0.7:
                                v_wall_positions.append(x)

                    # Group adjacent positions to handle thick walls
                    v_walls = []
                    if v_wall_positions:
                        current_wall = [v_wall_positions[0]]
                        for pos in v_wall_positions[1:]:
                            if pos - current_wall[-1] <= 3:  # Adjust threshold as needed
                                current_wall.append(pos)
                            else:
                                v_walls.append(sum(current_wall) / len(current_wall))  # Average position
                                current_wall = [pos]
                        if current_wall:
                            v_walls.append(sum(current_wall) / len(current_wall))  # Add last wall

                    # Assign walls to left or right edge based on position
                    for wall_pos in v_walls:
                        if wall_pos < cell.shape[1] / 2:
                            wall_indicators[r][c][3] = 1  # Left wall
                            if self.debug_mode:
                                x_pos = int(left + wall_pos)
                                cv2.line(debug_img, (x_pos, top), (x_pos, bottom), (0, 0, 255), 1)
                        else:
                            wall_indicators[r][c][1] = 1  # Right wall
                            if self.debug_mode:
                                x_pos = int(left + wall_pos)
                                cv2.line(debug_img, (x_pos, top), (x_pos, bottom), (0, 0, 255), 1)

            if self.debug_mode:
                cv2.imwrite("debug/05a_cell_analysis.png", debug_img)

            # Second pass: ensure wall consistency between adjacent cells
            for r in range(rows):
                for c in range(cols):
                    # Right-left consistency
                    if c < cols - 1:
                        if wall_indicators[r][c][1] == 1:  # Right wall
                            wall_indicators[r][c + 1][3] = 1  # Left wall for cell to the right
                        elif wall_indicators[r][c + 1][3] == 1:  # Left wall for cell to the right
                            wall_indicators[r][c][1] = 1  # Right wall

                    # Bottom-top consistency
                    if r < rows - 1:
                        if wall_indicators[r][c][2] == 1:  # Bottom wall
                            wall_indicators[r + 1][c][0] = 1  # Top wall for cell below
                        elif wall_indicators[r + 1][c][0] == 1:  # Top wall for cell below
                            wall_indicators[r][c][2] = 1  # Bottom wall

            if self.debug_mode:
                cv2.imwrite("debug_walls.png", debug_img)

            # Third pass: Handle border walls while preserving entry/exit points
            # Count openings in each border to identify potential entry/exit
            top_openings = [c for c in range(cols) if wall_indicators[0][c][0] == 0]
            right_openings = [r for r in range(rows) if wall_indicators[r][cols - 1][1] == 0]
            bottom_openings = [c for c in range(cols) if wall_indicators[rows - 1][c][2] == 0]
            left_openings = [r for r in range(rows) if wall_indicators[r][0][3] == 0]

            # Log the openings found
            if self.debug_mode:
                logging.debug(
                    f"Border openings - Top: {len(top_openings)}, Right: {len(right_openings)}, Bottom: {len(bottom_openings)}, Left: {len(left_openings)}")

            # Fill in top border walls (preserving potential entry/exit)
            preserve_top = []
            if len(top_openings) <= 2:  # Reasonable number of openings
                preserve_top = top_openings
            for c in range(cols):
                if c not in preserve_top:
                    wall_indicators[0][c][0] = 1  # Add top wall

            # Fill in right border walls (preserving potential entry/exit)
            preserve_right = []
            if len(right_openings) <= 2:  # Reasonable number of openings
                preserve_right = right_openings
            for r in range(rows):
                if r not in preserve_right:
                    wall_indicators[r][cols - 1][1] = 1  # Add right wall

            # Fill in bottom border walls (preserving potential entry/exit)
            preserve_bottom = []
            if len(bottom_openings) <= 2:  # Reasonable number of openings
                preserve_bottom = bottom_openings
            for c in range(cols):
                if c not in preserve_bottom:
                    wall_indicators[rows - 1][c][2] = 1  # Add bottom wall

            # Fill in left border walls (preserving potential entry/exit)
            preserve_left = []
            if len(left_openings) <= 2:  # Reasonable number of openings
                preserve_left = left_openings
            for r in range(rows):
                if r not in preserve_left:
                    wall_indicators[r][0][3] = 1  # Add left wall

            # Save the entry/exit points for later use if needed
            if hasattr(self, 'entry_exit_points'):
                # Find entry (typically on top or left border)
                entry_candidates = []
                if preserve_top:
                    entry_candidates.extend([(0, c) for c in preserve_top])
                if preserve_left:
                    entry_candidates.extend([(r, 0) for r in preserve_left])

                # Find exit (typically on bottom or right border)
                exit_candidates = []
                if preserve_bottom:
                    exit_candidates.extend([(rows - 1, c) for c in preserve_bottom])
                if preserve_right:
                    exit_candidates.extend([(r, cols - 1) for r in preserve_right])

                # Use closest to origin as entry, furthest from origin as exit
                if entry_candidates:
                    entry_candidates.sort(key=lambda pos: pos[0] + pos[1])
                    entry = entry_candidates[0]
                else:
                    entry = (0, 0)  # Default

                if exit_candidates:
                    exit_candidates.sort(key=lambda pos: pos[0] + pos[1], reverse=True)
                    exit_point = exit_candidates[0]
                else:
                    exit_point = (rows - 1, cols - 1)  # Default

                self.entry_exit_points = (entry, exit_point)

            # Debug visualization of final walls
            if self.debug_mode:
                border_debug = np.ones((rows * 50, cols * 50, 3), dtype=np.uint8) * 255
                for r in range(rows):
                    for c in range(cols):
                        cell_y, cell_x = r * 50, c * 50
                        for wall_dir in range(4):
                            if wall_indicators[r][c][wall_dir] == 1:
                                if wall_dir == 0:  # Top
                                    cv2.line(border_debug, (cell_x, cell_y), (cell_x + 50, cell_y), (0, 0, 0), 2)
                                elif wall_dir == 1:  # Right
                                    cv2.line(border_debug, (cell_x + 50, cell_y), (cell_x + 50, cell_y + 50), (0, 0, 0),
                                             2)
                                elif wall_dir == 2:  # Bottom
                                    cv2.line(border_debug, (cell_x, cell_y + 50), (cell_x + 50, cell_y + 50), (0, 0, 0),
                                             2)
                                elif wall_dir == 3:  # Left
                                    cv2.line(border_debug, (cell_x, cell_y), (cell_x, cell_y + 50), (0, 0, 0), 2)

                cv2.imwrite("debug/05b_border_handling.png", border_debug)

            return wall_indicators

        def _visualize_walls(self, wall_indicators, rows, cols):
            """
            Create a visualization of the detected walls.

            Args:
                wall_indicators: 3D list of wall indicators
                rows: Number of rows
                cols: Number of columns
            """
            # Create a debug grid image
            viz_cell_size = 50  # Size for visualization
            debug_grid = np.ones((rows * viz_cell_size, cols * viz_cell_size, 3), dtype=np.uint8) * 255

            for r in range(rows):
                for c in range(cols):
                    cell_walls = wall_indicators[r][c]

                    cell_y, cell_x = r * viz_cell_size, c * viz_cell_size
                    # Draw the cell
                    cv2.rectangle(debug_grid, (cell_x, cell_y), (cell_x + viz_cell_size, cell_y + viz_cell_size),
                                  (200, 200, 200), 1)

                    # Draw detected walls
                    if cell_walls[0]:  # Top
                        cv2.line(debug_grid, (cell_x, cell_y), (cell_x + viz_cell_size, cell_y), (0, 0, 0), 2)
                    if cell_walls[1]:  # Right
                        cv2.line(debug_grid, (cell_x + viz_cell_size, cell_y),
                                 (cell_x + viz_cell_size, cell_y + viz_cell_size), (0, 0, 0), 2)
                    if cell_walls[2]:  # Bottom
                        cv2.line(debug_grid, (cell_x, cell_y + viz_cell_size),
                                 (cell_x + viz_cell_size, cell_y + viz_cell_size), (0, 0, 0), 2)
                    if cell_walls[3]:  # Left
                        cv2.line(debug_grid, (cell_x, cell_y), (cell_x, cell_y + viz_cell_size), (0, 0, 0), 2)

                    # Add labels
                    cv2.putText(debug_grid, f"{r},{c}", (cell_x + 15, cell_y + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

            cv2.imwrite("debug/05_wall_detection.png", debug_grid)

    processor = MazeProcessor()

    # --- Utility function to compute cell index ---
    def index(i, j):
        if i < 0 or j < 0 or i >= cols or j >= rows:
            return None
        return i + j * cols

    # --- Redraw Maze on Canvas ---
    def redraw_maze():
        surface = pygame.Surface((width, height))
        surface.fill((255, 255, 255))
        for cell in grid:
            cell.show(surface, is_start=(cell == start), is_goal=(cell == goal))
        img_data = pygame.image.tostring(surface, "RGB")
        img = Image.frombytes("RGB", (width, height), img_data)
        img_tk = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.img = img_tk

    def load_maze_image():
        global cols, rows, grid, w, x_offset, y_offset, start, goal, start_time
        file_path = filedialog.askopenfilename(title="Select Maze Image")
        start_time = time.time()
        if not file_path:
            return
        try:

            # Process the image to create a grid of Cell objects
            cells, grid_rows, grid_cols = processor.process_image(file_path, grid_size=grid_size)
            grid.clear()
            grid.extend(cells)
            cols = grid_cols
            rows = grid_rows
            # Compute cell size for drawing based on the canvas dimensions
            w = min(width // cols, height // rows)
            maze_width = cols * w
            maze_height = rows * w
            x_offset = (width - maze_width) // 2
            y_offset = (height - maze_height) // 2
            print(cols, rows, w, maze_height, maze_width, x_offset, y_offset)

            # Find entrance and exit points (cells on the border with an open wall)
            border_openings = []
            for cell in grid:
                is_border = cell.i == 0 or cell.j == 0 or cell.i == cols - 1 or cell.j == rows - 1
                if is_border:
                    has_opening = False
                    if cell.i == 0 and not cell.walls[3]:
                        has_opening = True
                    elif cell.i == cols - 1 and not cell.walls[1]:
                        has_opening = True
                    elif cell.j == 0 and not cell.walls[0]:
                        has_opening = True
                    elif cell.j == rows - 1 and not cell.walls[2]:
                        has_opening = True
                    if has_opening:
                        border_openings.append(cell)

            # If exactly two openings are found, set them as start and goal
            if len(border_openings) == 2:
                start, goal = border_openings
            # If more than two, choose the pair with maximum distance
            elif len(border_openings) > 2:
                max_distance = 0
                start = goal = None
                for i, cell1 in enumerate(border_openings):
                    for cell2 in border_openings[i + 1:]:
                        distance = abs(cell1.i - cell2.i) + abs(cell1.j - cell2.j)
                        if distance > max_distance:
                            max_distance = distance
                            start, goal = cell1, cell2
            elif len(border_openings) == 1:
                start = border_openings[0]
                # Find the farthest border cell from start
                max_distance = 0
                goal = None
                # List of border cells
                border_cells = [
                    cell for cell in grid if cell.i == 0 or cell.i == cols - 1 or cell.j == 0 or cell.j == rows - 1
                ]
                for cell in border_cells:
                    distance = abs(cell.i - start.i) + abs(cell.j - start.j)  # Manhattan distance
                    if distance > max_distance:
                        max_distance = distance
                        goal = cell
                if goal:
                    # Open the appropriate wall based on the goal's position
                    if goal.i == 0:  # Left column
                        goal.walls[3] = False  # Open Left wall
                    elif goal.i == cols - 1:  # Right column
                        goal.walls[1] = False  # Open right wall
                    elif goal.j == 0:  # Top row
                        goal.walls[0] = False  # Open top wall
                    elif goal.j == rows - 1:  # Bottom row
                        goal.walls[2] = False  # Open bottom wall
            else:
                # If no openings detected, force openings at opposite corners
                print("No openings detected in maze border. Creating openings at opposite corners.")
                start = next((cell for cell in grid if cell.i == 0 and cell.j == 0), None)
                goal = next((cell for cell in grid if cell.i == cols - 1 and cell.j == rows - 1), None)
                if start:
                    start.walls[3] = False
                if goal:
                    goal.walls[1] = False

            redraw_maze()
            # Calculate and update execution time
            end_time = time.time()
            execution_time = round(end_time - start_time, 2)
            execution_time_label.config(text=f"Execution Time: {execution_time}s")
            messagebox.showinfo("Maze Loaded", "Maze loaded from image successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load maze image: {str(e)}")

    # --- Update GUI during solving ---
    def update_gui(path, solving, surface):
        surface.fill((255, 255, 255))
        for cell in grid:
            cell.show(surface, is_start=(cell == start), is_goal=(cell == goal))
        if solving:
            for i in range(len(path) - 1):
                x1 = path[i].i * w + x_offset + w // 2
                y1 = path[i].j * w + y_offset + w // 2
                x2 = path[i + 1].i * w + x_offset + w // 2
                y2 = path[i + 1].j * w + y_offset + w // 2
                pygame.draw.line(surface, (0, 0, 255), (x1, y1), (x2, y2), 2)
        else:
            final_path = path + [goal]
            for i in range(len(final_path) - 1):
                x1 = final_path[i].i * w + x_offset + w // 2
                y1 = final_path[i].j * w + y_offset + w // 2
                x2 = final_path[i + 1].i * w + x_offset + w // 2
                y2 = final_path[i + 1].j * w + y_offset + w // 2
                pygame.draw.line(surface, (255, 0, 0), (x1, y1), (x2, y2), 3)
            img_data = pygame.image.tostring(surface, "RGB")
            img = Image.frombytes("RGB", (width, height), img_data)
            img_tk = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            canvas.img = img_tk
            messagebox.showinfo("Maze Solved", "Maze solved")
            return
        img_data = pygame.image.tostring(surface, "RGB")
        img = Image.frombytes("RGB", (width, height), img_data)
        img_tk = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.img = img_tk

    # --- Solve Maze using Selected Algorithm ---
    def solve_maze_selected():
        if algorithm_var.get() == "BFS":
            solve_maze_bfs(start, goal, grid, index, update_gui, canvas, execution_time_label, root, width, height)
        elif algorithm_var.get() == "A*":
            solve_maze_a_star(start, goal, grid, index, update_gui, canvas, execution_time_label, root, width, height)
        elif algorithm_var.get() == "DFS":
            solve_maze_dfs(start, goal, grid, index, update_gui, canvas, execution_time_label, root, width, height)
        elif algorithm_var.get() == "Dijkstra":
            solve_maze_dijkstra(start, goal, grid, index, canvas, execution_time_label, root, width, height, w,
                                x_offset,
                                y_offset)

    # --- Reset Maze (regenerate with current size) ---
    def reset_maze():
        surface = pygame.Surface((width, height))
        surface.fill((255, 255, 255))
        for cell in grid:
            cell.show(surface, is_start=(cell == start), is_goal=(cell == goal))
        img_data = pygame.image.tostring(surface, "RGB")
        img = Image.frombytes("RGB", (width, height), img_data)
        img_tk = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.img = img_tk
        execution_time_label.config(text="Execution Time: 0s")

    # --- Clear Maze (remove current maze and reset inputs) ---
    def clear_maze():
        global cols, rows, grid, stack, current, start, goal
        cols = rows = 0
        grid.clear()
        stack.clear()
        start = goal = None
        canvas.delete("all")
        execution_time_label.config(text="Execution Time: 0s")

    # Replace the root window with the provided parent frame
    root = parent_frame

    # Initialize GUI components
    top_frame = tk.Frame(root)
    top_frame.pack()

    top_frame = tk.Frame(root, bg="#d3d3d3", padx=10, pady=10)
    top_frame.pack(side=tk.TOP, fill=tk.X)

    execution_time_label = tk.Label(root, text="Execution Time: 0.0s", font=('Arial', 12))
    execution_time_label.pack(pady=10)

    load_image_button = tk.Button(top_frame, text="Load Maze Image", command=load_maze_image)
    load_image_button.pack(side=tk.LEFT, padx=5)
    clear_button = tk.Button(top_frame, text="Clear Maze", command=clear_maze)
    clear_button.pack(side=tk.LEFT, padx=5)
    reset_button = tk.Button(top_frame, text="Reset Maze", command=reset_maze)
    reset_button.pack(side=tk.LEFT, padx=5)

    # Button to solve maze using the selected algorithm
    solve_button = tk.Button(top_frame, text="Solve Maze", command=solve_maze_selected)
    solve_button.pack(side=tk.RIGHT, padx=10)

    # Algorithm selection radio buttons
    algo_frame = tk.Frame(top_frame, bg="#d3d3d3")
    algo_frame.pack(side=tk.RIGHT)
    algorithm_var = tk.StringVar(value="BFS")
    astar_radio = tk.Radiobutton(algo_frame, text="A*", variable=algorithm_var, value="A*", bg="#d3d3d3")
    astar_radio.pack(side=tk.RIGHT, padx=5)
    dfs_radio = tk.Radiobutton(algo_frame, text="DFS", variable=algorithm_var, value="DFS", bg="#d3d3d3")
    dfs_radio.pack(side=tk.RIGHT, padx=5)
    bfs_radio = tk.Radiobutton(algo_frame, text="BFS", variable=algorithm_var, value="BFS", bg="#d3d3d3")
    bfs_radio.pack(side=tk.RIGHT, padx=5)
    dijkstra_radio = tk.Radiobutton(algo_frame, text="Dijkstra", variable=algorithm_var, value="Dijkstra", bg="#d3d3d3")
    dijkstra_radio.pack(side=tk.RIGHT, padx=5)
    algo_label = tk.Label(top_frame, text="Algorithm:", bg="#d3d3d3")
    algo_label.pack(side=tk.RIGHT, padx=5)

    canvas = tk.Canvas(root, width=600, height=600)
    canvas.pack()
