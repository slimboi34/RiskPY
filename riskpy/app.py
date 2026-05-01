import tkinter as tk
from tkinter import ttk, filedialog
import csv
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .cpp_underwriter import RiskEngine, ExcelExporter, Field, FactorModel, ActuarialMath, MonteCarloSimulator

class UnderwritingApp:
    def __init__(self, title="Actuarial Underwriter", excel_template="corporate_layout.xlsx"):
        self.title = title
        self.engine = RiskEngine()
        self.exporter = ExcelExporter(template=excel_template)
        self.engine.attach_exporter(self.exporter)
        
        self.fields_config = [] # stores dicts: {name, label, col, choices}

    def add_field(self, name, label, excel_col, choices=None):
        """Adds a parameter to the UI, connects it to C++ Engine, and maps it to Excel"""
        field_type = "string" if choices else "float"
        self.engine.add_field(Field(name=name, label=label, type=field_type))
        self.exporter.map_column(excel_col, name, label)
        self.fields_config.append({"name": name, "label": label, "choices": choices})

    def set_logic(self, logic_func):
        """Sets the python mathematical logic callback"""
        self.engine.set_math_logic(logic_func)

    def set_factor_model(self, factor_model: FactorModel):
        """Replaces manual Python callbacks with the hyper-fast C++ Factor Rating Engine"""
        self.engine.set_math_logic(factor_model.calculate)

    def set_premium_column(self, excel_col, label="Final Premium"):
        """Defines where the final calculated answer is exported in Excel"""
        self.exporter.map_column(excel_col, "premium_output", label)

    def calculate_headless(self, inputs: dict) -> float:
        """Runs calculation directly without GUI"""
        return self.engine.execute(inputs)
        
    def export_excel_headless(self, filename="policy_quote.xlsx"):
        """Exports the last calculation to excel"""
        self.engine.export_to_excel(filename)

    def calculate_batch(self, csv_filepath, output_filename="batch_policy_quotes.xlsx"):
        total_premium = 0.0
        success_count = 0
        batch_inputs = []
        batch_premiums = []
        with open(csv_filepath, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                parsed_inputs = {}
                for k, v in row.items():
                    try:
                        parsed_inputs[k] = float(v)
                    except ValueError:
                        parsed_inputs[k] = str(v)
                
                prem = self.engine.execute(parsed_inputs)
                batch_inputs.append(parsed_inputs)
                batch_premiums.append(prem)
                total_premium += prem
                success_count += 1
                
        self.engine.export_batch_to_excel(output_filename, batch_inputs, batch_premiums)
        return total_premium, success_count

    def _render_simulation_tab(self, parent):
        """Renders the Monte Carlo tail risk simulator in the GUI"""
        frame = ttk.Frame(parent)
        
        # Inputs
        input_frame = ttk.Frame(frame)
        input_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(input_frame, text="Trials:").grid(row=0, column=0)
        e_trials = tk.Entry(input_frame, width=10); e_trials.insert(0, "10000")
        e_trials.grid(row=0, column=1)
        
        tk.Label(input_frame, text="Exp. Claim Freq (λ):").grid(row=0, column=2)
        e_freq = tk.Entry(input_frame, width=10); e_freq.insert(0, "5.0")
        e_freq.grid(row=0, column=3)
        
        tk.Label(input_frame, text="Severity Base (μ):").grid(row=0, column=4)
        e_mu = tk.Entry(input_frame, width=10); e_mu.insert(0, "10.0")
        e_mu.grid(row=0, column=5)
        
        tk.Label(input_frame, text="Volatility (σ):").grid(row=0, column=6)
        e_sigma = tk.Entry(input_frame, width=10); e_sigma.insert(0, "1.5")
        e_sigma.grid(row=0, column=7)

        # Plot Area
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=10)

        def run_sim():
            trials = int(e_trials.get())
            freq = float(e_freq.get())
            mu = float(e_mu.get())
            sigma = float(e_sigma.get())

            sim = MonteCarloSimulator(trials=trials)
            losses = sim.simulate_aggregate_loss(freq, mu, sigma)
            
            # Rendering
            ax.clear()
            ax.hist(losses, bins=50, color='blue', alpha=0.7)
            ax.set_title(f"Stochastic Aggregate Loss ({trials:,} Trials)")
            ax.set_xlabel("Total Simulated Loss ($)")
            ax.set_ylabel("Frequency")
            
            # Tail risk line
            var_99 = np.percentile(losses, 99)
            ax.axvline(var_99, color='red', linestyle='dashed', linewidth=2, label=f'99% VaR: ${var_99:,.0f}')
            ax.legend()
            
            canvas.draw()

        tk.Button(frame, text="Run C++ Monte Carlo", command=run_sim, font=("Arial", 12, "bold"), fg="purple").pack(pady=10)
        return frame

    def run(self, export_filename="policy_quote.xlsx"):
        """Spins up the Tkinter GUI and maps all data"""
        root = tk.Tk()
        root.title(self.title)
        root.geometry("800x600")
        
        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)
        
        # TAB 1: Rating Engine
        rating_frame = ttk.Frame(notebook)
        notebook.add(rating_frame, text="Pricing & Rating")
        
        tk_inputs = {}
        row_idx = 0
        for config in self.fields_config:
            tk.Label(rating_frame, text=config['label'], font=("Arial", 12)).grid(row=row_idx, column=0, padx=10, pady=10, sticky="w")
            
            if config['choices']:
                entry = ttk.Combobox(rating_frame, values=config['choices'], font=("Arial", 12), state="readonly")
                entry.current(0)
            else:
                entry = tk.Entry(rating_frame, font=("Arial", 12))
                
            entry.grid(row=row_idx, column=1, padx=10, pady=10)
            tk_inputs[config['name']] = entry
            row_idx += 1

        def on_submit():
            dynamic_inputs = {}
            for fname, entry_widget in tk_inputs.items():
                val = entry_widget.get()
                try:
                    dynamic_inputs[fname] = float(val)
                except ValueError:
                    dynamic_inputs[fname] = str(val)
                    
            result = self.engine.execute(dynamic_inputs) 
            self.engine.export_to_excel(export_filename) 
            result_label.config(text=f"Premium Calculated: ${result:,.2f}\nExported to {export_filename}")

        def on_batch():
            filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
            if filepath:
                total_premium, count = self.calculate_batch(filepath)
                result_label.config(text=f"Batch processed {count} records.\nTotal Book Premium: ${total_premium:,.2f}\nExported to batch_policy_quotes.xlsx")

        submit_btn = tk.Button(rating_frame, text="Calculate Profile", font=("Arial", 12, "bold"), fg="black", command=on_submit)
        submit_btn.grid(row=row_idx, column=0, pady=20)
        
        batch_btn = tk.Button(rating_frame, text="Run CSV Batch", font=("Arial", 12, "bold"), fg="blue", command=on_batch)
        batch_btn.grid(row=row_idx, column=1, pady=20)
        
        row_idx += 1
        result_label = tk.Label(rating_frame, text="", font=("Arial", 12, "bold"), fg="#333333")
        result_label.grid(row=row_idx, column=0, columnspan=2)

        # TAB 2: Predictive Simulation
        sim_frame = self._render_simulation_tab(notebook)
        notebook.add(sim_frame, text="Monte Carlo Predictor")
        
        root.mainloop()
