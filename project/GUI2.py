import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time

class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Viewer")

        # Variables
        self.selected_option = tk.StringVar()
        self.file_path = tk.StringVar()
        self.youtube_link = tk.StringVar()

        # GUI components
        self.create_dropdown_menu()
        self.create_input_area()
        self.create_buttons()
        self.create_image_display()

    def create_dropdown_menu(self):
        options = ["youtube link", "datei pfad"]
        dropdown_label = ttk.Label(self.root, text="Select Option:")
        dropdown_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)

        dropdown_menu = ttk.Combobox(self.root, textvariable=self.selected_option, values=options)
        dropdown_menu.grid(row=0, column=1, padx=10, pady=10, sticky=tk.W)
        dropdown_menu.current(0)

    def create_input_area(self):
        input_label = ttk.Label(self.root, text="Input:")
        input_label.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)

        entry_frame = ttk.Frame(self.root)
        entry_frame.grid(row=1, column=1, padx=10, pady=10, sticky=tk.W)

        ttk.Entry(entry_frame, textvariable=self.file_path).grid(row=0, column=0)
        ttk.Entry(entry_frame, textvariable=self.youtube_link).grid(row=0, column=1)

    def create_buttons(self):
        read_button = ttk.Button(self.root, text="Einlesen", command=self.read_input)
        read_button.grid(row=1, column=2, padx=10, pady=10, sticky=tk.W)

        analyze_button = ttk.Button(self.root, text="Auswertung", command=self.start_analysis)
        analyze_button.grid(row=0, column=2, padx=10, pady=10, sticky=tk.W)

    def create_image_display(self):
        image_frame_1 = ttk.Frame(self.root)
        image_frame_1.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
        self.image_label_1 = tk.Label(image_frame_1)
        self.image_label_1.pack()

        graph_frame_1 = ttk.Frame(self.root)
        graph_frame_1.grid(row=3, column=0, padx=10, pady=10, sticky=tk.W)
        self.plot_graph(graph_frame_1)

        image_frame_2 = ttk.Frame(self.root)
        image_frame_2.grid(row=2, column=1, padx=10, pady=10, sticky=tk.W)
        self.image_label_2 = tk.Label(image_frame_2)
        self.image_label_2.pack()

        graph_frame_2 = ttk.Frame(self.root)
        graph_frame_2.grid(row=3, column=1, padx=10, pady=10, sticky=tk.W)
        self.plot_graph(graph_frame_2)

    def read_input(self):
        if self.selected_option.get() == "datei pfad":
            file_path = filedialog.askopenfilename()
            self.file_path.set(file_path)

    def start_analysis(self):
        # Call your function.py function here
        # For example:
        # thread = threading.Thread(target=function.analysis_function, args=(self.file_path.get(), self.youtube_link.get()))
        # thread.start()

        # Simulate image updating and plotting for demonstration
        thread = threading.Thread(target=self.simulate_analysis)
        thread.start()

    def simulate_analysis(self):
        while True:
            # Simulate updating images and getting data for plotting
            image_1, image_2 = update_images()

            # Update image labels
            self.update_image_label(self.image_label_1, image_1)
            self.update_image_label(self.image_label_2, image_2)

            # Get data for plotting
            data_1 = get_data_for_plot()
            data_2 = get_data_for_plot()

            # Plot graphs
            self.plot_graph_data(data_1, 0)
            self.plot_graph_data(data_2, 1)

            # Pause for a while (simulating processing time)
            time.sleep(1)

    def update_image_label(self, label, image):
        # Update the image label with the new image
        # You need to implement the logic to update the image label based on your requirements
        # For example, you can use PIL or other libraries to work with images
        # label.configure(image=new_image)
        pass

    def plot_graph(self, frame):
        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack()
        canvas.draw()

    def plot_graph_data(self, data, index):
        # Plot the data on the corresponding graph
        # You need to implement the logic to update the graph based on your requirements
        # For example, use ax.plot() to update the graph
        pass


