import tkinter as tk
from tkinter import ttk

window = tk.Tk()
window.title('PyVAWT')
window.geometry('600x400')

# Frame

turbine_config = ttk.Frame(window)
turbine_config.pack()

environment_config = ttk.Frame(window)
environment_config.pack()

# Turbine config frame widgets

turbine_label = ttk.Label(turbine_config, text='Turbine Parameters', font=('Arial', 12, 'bold'))
turbine_label.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky='w')

r_label = ttk.Label(turbine_config, text='Radius:')
r_label.grid(row=1, column=0, padx=5, pady=5, sticky='e')
r_entry = ttk.Entry(turbine_config)
r_entry.grid(row=1, column=1, padx=5, pady=5)

H_label = ttk.Label(turbine_config, text='Height:')
H_label.grid(row=2, column=0, padx=5, pady=5, sticky='e')
H_entry = ttk.Entry(turbine_config)
H_entry.grid(row=2, column=1, padx=5, pady=5)

twist_label = ttk.Label(turbine_config, text='Twist:')
twist_label.grid(row=3, column=0, padx=5, pady=5, sticky='e')
twist_entry = ttk.Entry(turbine_config)
twist_entry.grid(row=3, column=1, padx=5, pady=5)

delta_label = ttk.Label(turbine_config, text='Delta:')
delta_label.grid(row=4, column=0, padx=5, pady=5, sticky='e')
delta_entry = ttk.Entry(turbine_config)
delta_entry.grid(row=4, column=1, padx=5, pady=5)

chord_label = ttk.Label(turbine_config, text='Chord:')
chord_label.grid(row=5, column=0, padx=5, pady=5, sticky='e')
chord_entry = ttk.Entry(turbine_config)
chord_entry.grid(row=5, column=1, padx=5, pady=5)

B_label = ttk.Label(turbine_config, text='Blades:')
B_label.grid(row=6, column=0, padx=5, pady=5, sticky='e')
B_entry = ttk.Entry(turbine_config)
B_entry.grid(row=6, column=1, padx=5, pady=5)

sol_label = ttk.Label(turbine_config, text='Solidity:')
sol_label.grid(row=7, column=0, padx=5, pady=5, sticky='e')
sol_entry = ttk.Entry(turbine_config)
sol_entry.grid(row=7, column=1, padx=5, pady=5)

omega_label = ttk.Label(turbine_config, text='Omega:')
omega_label.grid(row=8, column=0, padx=5, pady=5, sticky='e')
omega_entry = ttk.Entry(turbine_config)
omega_entry.grid(row=8, column=1, padx=5, pady=5)

ntheta_label = ttk.Label(turbine_config, text='Ntheta:')
ntheta_label.grid(row=9, column=0, padx=5, pady=5, sticky='e')
ntheta_entry = ttk.Entry(turbine_config)
ntheta_entry.grid(row=9, column=1, padx=5, pady=5)

# Environment config frame widgets

environment_label = ttk.Label(environment_config, text='Environment Parameters', font=('Arial', 12, 'bold'))
environment_label.grid(row=0, column=0, sticky='w')

# run
window.mainloop()