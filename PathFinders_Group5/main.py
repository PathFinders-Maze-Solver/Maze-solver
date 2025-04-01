import tkinter as tk
import Input_maze_size
import Load_maze_image


# Function to load algo.py content
def load_algo():
    # Clear the current content
    for widget in content_frame.winfo_children():
        widget.destroy()

    # Show the back button in the title bar
    back_button.pack(side=tk.RIGHT, padx=10, pady=5)

    # Load algo.py content
    algo_frame = tk.Frame(content_frame)
    algo_frame.pack(fill=tk.BOTH, expand=True)

    # Add algo.py functionality to the frame
    Input_maze_size.setup(algo_frame)  # Call a setup function from algo.py to initialize its content


# Function to load vision.py content
def load_vision():
    # Clear the current content
    for widget in content_frame.winfo_children():
        widget.destroy()

    # Show the back button in the title bar
    back_button.pack(side=tk.RIGHT, padx=10, pady=5)

    # Load vision.py content
    vision_frame = tk.Frame(content_frame)
    vision_frame.pack(fill=tk.BOTH, expand=True)

    # Add vision.py functionality to the frame
    Load_maze_image.setup(vision_frame)  # Call a setup function from vision.py to initialize its content


# Function to change button color on hover
def on_enter(e):
    e.widget.config(bg="#006d77", fg="white")  # Change color on hover


def on_leave(e):
    e.widget.config(bg="#006d77", fg="black")  # Revert back


# Function to show the main menu (selection screen)
def show_main_menu():
    # Clear the current content
    for widget in content_frame.winfo_children():
        widget.destroy()

    # Hide the back button in the title bar
    back_button.pack_forget()
    # Welcome text style
    welcome_style = {
        "font": ("Arial", 16, "bold"),
        "fg": "#005",  # Peacock blue text color
        "pady": 20  # Add some padding at the top
    }

    # Add welcome text
    welcome_label = tk.Label(content_frame, text="Welcome to the PathFinders Maze Puzzle Solver!", **welcome_style)
    welcome_label.pack(pady=20)  # Add padding below the welcome text

    # Button style
    button_style = {
        "width": 30, "height": 2,
        "font": ("Arial", 12, "bold"),
        "bg": "#006d77", "fg": "white",  # Peacock blue background, white text
        "bd": 0, "relief": "flat",
        "activebackground": "#0097a7",  # Lighter teal for hover
        "activeforeground": "white"
    }

    # Create buttons with hover effects
    algo_button = tk.Button(content_frame, text="Input Maze Size", command=load_algo, **button_style)
    vision_button = tk.Button(content_frame, text="Load Maze Image", command=load_vision, **button_style)

    algo_button.bind("<Enter>", on_enter)
    algo_button.bind("<Leave>", on_leave)
    vision_button.bind("<Enter>", on_enter)
    vision_button.bind("<Leave>", on_leave)

    # Pack buttons with spacing
    algo_button.pack(pady=20, ipadx=10, ipady=5)
    vision_button.pack(pady=20, ipadx=10, ipady=5)


# Function to close the application
def close_app():
    root.destroy()


# Create the main window
root = tk.Tk()
root.title("Maze Solver")
root.geometry("800x600")  # Set window size

# Title and Group Number Frame
title_frame = tk.Frame(root, bg="#333", padx=10, pady=10)
title_frame.pack(fill=tk.X)

# Title and Group Number
title_label = tk.Label(title_frame, text="PathFinders - Group Number 5",
                       font=("Arial", 20, "bold"), fg="white", bg="#333")
title_label.place(relx=0.5, rely=0.5, anchor="center")  # Center the title

# Back Button (top-right, before close button)
back_button = tk.Button(title_frame, text="⬅ Back", command=show_main_menu, font=("Arial", 12), bg="lightgray",
                        fg="black", bd=0)

# Close Button (top-right)
close_button = tk.Button(title_frame, text="", command=close_app, font=("Arial", 14), bg="#333", bd=0)
close_button.pack(side=tk.RIGHT, padx=10, pady=5)

# Content Frame (for dynamic content)
content_frame = tk.Frame(root)
content_frame.pack(fill=tk.BOTH, expand=True)

# Show the main menu initially
show_main_menu()

# Run the Tkinter main loop
root.mainloop()
