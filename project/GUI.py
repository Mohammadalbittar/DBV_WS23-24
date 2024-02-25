import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class GUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Image Analysis GUI")

        # Dropdown menu
        self.options = ["YouTube Link", "Datei Pfad"]
        self.selected_option = tk.StringVar(value=self.options[0])

        self.dropdown_menu = ttk.Combobox(self, values=self.options, textvariable=self.selected_option)
        self.dropdown_menu.grid(row=0, column=0, padx=10, pady=10)

        # Auswertung button
        self.auswertung_button = tk.Button(self, text="Auswertung", command=self.start_analysis)
        self.auswertung_button.grid(row=0, column=1, padx=10, pady=10)

        # Image displays
        self.image1_label = tk.Label(self)
        self.image1_label.grid(row=1, column=0, padx=10, pady=10)

        self.image2_label = tk.Label(self)
        self.image2_label.grid(row=1, column=1, padx=10, pady=10)

        # Matplotlib graphs
        self.figure1, self.ax1 = plt.subplots()
        self.canvas1 = FigureCanvasTkAgg(self.figure1, master=self)
        self.canvas1_widget = self.canvas1.get_tk_widget()
        self.canvas1_widget.grid(row=2, column=0, padx=10, pady=10)

        self.figure2, self.ax2 = plt.subplots()
        self.canvas2 = FigureCanvasTkAgg(self.figure2, master=self)
        self.canvas2_widget = self.canvas2.get_tk_widget()
        self.canvas2_widget.grid(row=2, column=1, padx=10, pady=10)

    def start_analysis(self):
        # Call the function from function.py
        function.start_analysis(self)

    def update_images(self, image1_path, image2_path):
        # Load images
        img1 = Image.open(image1_path)
        img2 = Image.open(image2_path)

        # Resize images if needed
        img1 = img1.resize((300, 300), Image.ANTIALIAS)
        img2 = img2.resize((300, 300), Image.ANTIALIAS)

        # Update image labels
        self.image1_tk = ImageTk.PhotoImage(img1)
        self.image1_label.config(image=self.image1_tk)

        self.image2_tk = ImageTk.PhotoImage(img2)
        self.image2_label.config(image=self.image2_tk)

        # Call function to update Matplotlib graphs
        function.update_graphs(self)
