import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Sample data (replace with real data)
sample_data = {
    "Temperature Rise": {
        "years": [2000, 2005, 2010, 2015, 2020],
        "values": [0.5, 0.7, 0.9, 1.1, 1.3]
    },
    "CO2 Emissions": {
        "years": [2000, 2005, 2010, 2015, 2020],
        "values": [24, 28, 31, 34, 36]
    }
}

class ClimateChangeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Climate Change Dashboard")
        self.root.geometry("700x500")

        # Title Label
        title = tk.Label(root, text="🌍 Climate Change Dashboard", font=("Helvetica", 20, "bold"))
        title.pack(pady=10)

        # Dropdown for category
        self.category_var = tk.StringVar()
        category_label = tk.Label(root, text="Select Data Category:")
        category_label.pack()
        self.category_dropdown = ttk.Combobox(root, textvariable=self.category_var, state="readonly")
        self.category_dropdown['values'] = list(sample_data.keys())
        self.category_dropdown.pack(pady=5)

        # Button to show data
        show_button = tk.Button(root, text="Show Data", command=self.plot_data)
        show_button.pack(pady=10)

        # Placeholder for Matplotlib figure
        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_data(self):
        category = self.category_var.get()
        if category in sample_data:
            self.ax.clear()
            years = sample_data[category]["years"]
            values = sample_data[category]["values"]
            self.ax.plot(years, values, marker='o')
            self.ax.set_title(category)
            self.ax.set_xlabel("Year")
            self.ax.set_ylabel("Value")
            self.canvas.draw()

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = ClimateChangeGUI(root)
    root.mainloop()

import matplotlib.pyplot as plt

plt.plot([1, 2, 3, 4], [10, 20, 25, 30])
plt.title("Test Plot")
plt.show()
