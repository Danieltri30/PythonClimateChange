import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd
from data_processor import DataProcessor

class ClimateChangeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Climate Change Dashboard")
        self.root.geometry("750x550")

        # Create a frame for mode selection (Country vs City)
        mode_frame = tk.Frame(root)
        mode_label = tk.Label(mode_frame, text="Select Data Mode:")
        mode_label.pack(side=tk.LEFT, padx=(0, 10))

        # StringVar to track the mode: "country" or "city"
        self.mode_var = tk.StringVar(value="country")
        country_radio = tk.Radiobutton(
            mode_frame,
            text="Country",
            variable=self.mode_var,
            value="country",
            command=self.update_mode
        )
        country_radio.pack(side=tk.LEFT)
        city_radio = tk.Radiobutton(
            mode_frame,
            text="City",
            variable=self.mode_var,
            value="city",
            command=self.update_mode
        )
        city_radio.pack(side=tk.LEFT)
        mode_frame.pack(pady=10)

        # Dropdown for selecting country or city based on mode
        self.selection_var = tk.StringVar()
        selection_label = tk.Label(root, text="Select Option:")
        selection_label.pack()
        self.selection_dropdown = ttk.Combobox(root, textvariable=self.selection_var, state="readonly")
        self.selection_dropdown.pack(pady=5)

        # Button to show data plot for the selected country or city
        show_button = tk.Button(root, text="Show Data", command=self.plot_data)
        show_button.pack(pady=10)

        # Set up Matplotlib figure and canvas
        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Load both datasets
        self.load_data()
        # Update dropdown options based on the initial mode (default "country")
        self.update_mode()

    def load_data(self):
        # Load and clean country data
        dp_country = DataProcessor("../data/GlobalLandTemperaturesByCountry.csv")
        self.df_country = dp_country.clean_data()
        self.df_country["dt"] = pd.to_datetime(self.df_country["dt"])
        self.df_country["year"] = self.df_country["dt"].dt.year

        # Load and clean city data
        dp_city = DataProcessor("../data/GlobalLandTemperaturesByMajorCity.csv")
        self.df_city = dp_city.clean_data()
        self.df_city["dt"] = pd.to_datetime(self.df_city["dt"])
        self.df_city["year"] = self.df_city["dt"].dt.year

    def update_mode(self):
        """Update the dropdown options based on the mode selected."""
        mode = self.mode_var.get()
        if mode == "country":
            # Use unique country names from the country dataset
            options = sorted(self.df_country["Country"].unique())
        elif mode == "city":
            # Use unique city names from the city dataset
            options = sorted(self.df_city["City"].unique())
        self.selection_dropdown['values'] = options
        if options:
            self.selection_dropdown.current(0)

    def plot_data(self):
        mode = self.mode_var.get()
        selection = self.selection_var.get()

        # Depending on the mode, filter the appropriate dataframe
        if mode == "country":
            filtered_data = self.df_country[self.df_country["Country"] == selection]
        elif mode == "city":
            filtered_data = self.df_city[self.df_city["City"] == selection]

        if filtered_data.empty:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            self.canvas.draw()
            return

        # Group the filtered data by year to get average temperature per year
        yearly_avg = filtered_data.groupby("year")["AverageTemperature"].mean().reset_index()

        # Plot the data in the Tkinter embedded canvas
        self.ax.clear()
        self.ax.plot(yearly_avg["year"], yearly_avg["AverageTemperature"], marker="o")
        self.ax.set_title(f"Avg Temperature Over Time - {selection}")
        self.ax.set_xlabel("Year")
        self.ax.set_ylabel("Average Temperature")
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = ClimateChangeGUI(root)
    root.mainloop()
