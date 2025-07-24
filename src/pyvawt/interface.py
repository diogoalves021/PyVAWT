import tkinter as tk
import ttkbootstrap as ttk 

window = ttk.Window(themename= 'darkly')
window.title('PyVAWT')
window.geometry('700x600')
window.minsize('420','600')

# Frames

text_frame = ttk.Frame(window)
text_frame.pack()

turbine_config = ttk.Frame(window)
turbine_config.pack()

radio_buttons_frame = ttk.Frame(window)
radio_buttons_frame.pack(pady=20)

bottom_frame = ttk.Frame(window)
bottom_frame.pack()

# Functions

def run_from_gui():
    from .utils import load_config, save_config
    from .main import run_simulation

    config = load_config()

    config['turbine']['r'] = float(r_var.get())
    config['turbine']['H'] = float(H_var.get())
    config['turbine']['twist'] = float(twist_var.get())
    config['turbine']['delta'] = float(delta_var.get())
    config['turbine']['chord'] = [float(chord_var.get())]
    config['turbine']['B'] = int(B_var.get())
    config['turbine']['solidity'] = [float(sol_var.get())]
    config['turbine']['Omega'] = float(omega_var.get())
    config['turbine']['ntheta'] = int(ntheta_var.get())

    config['environment']['Vinf'] = [float(vinf_var.get())]
    config['simulation']['airfoil'] = [airfoil_var.get()]
    config['simulation']['num_turbines'] = 1
    config['simulation']['var_omega_vinf'] = 1

    config['simulation']['fixed_parameter'] = fixed_param_var.get()
    config['output']['save'] = bool(save_check_var.get())
    config['output']['save_config'] = bool(save_config_var.get())
    config['output']['save_plot'] = bool(save_plot_var.get())

    temp_path = 'src/pyvawt/config/gui_config.yaml'
    save_config(config, temp_path)

    run_simulation()


def get_info(params):
    for param in params:
        print(param.get())

# Title frame

title_label = ttk.Label(text_frame, text='Simulation Parameters', font=('Arial', 12, 'bold'))
title_label.pack()

info_label = ttk.Label(text_frame, text='Insert simulation parameters and click in the button to start simulating.', font=('Arial', 10))
info_label.pack(side='left')

# Turbine config frame widgets

r_var = tk.StringVar(value='3.0')
r_label = ttk.Label(turbine_config, text='Radius:')
r_label.grid(row=1, column=0, padx=5, pady=5, sticky='e')
r_entry = ttk.Entry(turbine_config, textvariable = r_var)
r_entry.grid(row=1, column=1, padx=5, pady=5)

H_var = tk.StringVar(value='1.0')
H_label = ttk.Label(turbine_config, text='Height:')
H_label.grid(row=2, column=0, padx=5, pady=5, sticky='e')
H_entry = ttk.Entry(turbine_config, textvariable=H_var)
H_entry.grid(row=2, column=1, padx=5, pady=5)

twist_var = tk.StringVar(value='0.0')
twist_label = ttk.Label(turbine_config, text='Twist:')
twist_label.grid(row=3, column=0, padx=5, pady=5, sticky='e')
twist_entry = ttk.Entry(turbine_config, textvariable=twist_var)
twist_entry.grid(row=3, column=1, padx=5, pady=5)

delta_var = tk.StringVar(value='0.0')
delta_label = ttk.Label(turbine_config, text='Delta:')
delta_label.grid(row=4, column=0, padx=5, pady=5, sticky='e')
delta_entry = ttk.Entry(turbine_config, textvariable=delta_var)
delta_entry.grid(row=4, column=1, padx=5, pady=5)

chord_var = tk.StringVar(value='0.25')
chord_label = ttk.Label(turbine_config, text='Chord:')
chord_label.grid(row=5, column=0, padx=5, pady=5, sticky='e')
chord_entry = ttk.Entry(turbine_config, textvariable=chord_var)
chord_entry.grid(row=5, column=1, padx=5, pady=5)

B_var = tk.StringVar(value='3')
B_label = ttk.Label(turbine_config, text='Blades:')
B_label.grid(row=6, column=0, padx=5, pady=5, sticky='e')
B_entry = ttk.Entry(turbine_config, textvariable=B_var)
B_entry.grid(row=6, column=1, padx=5, pady=5)

sol_var = tk.StringVar(value='0.125')
sol_label = ttk.Label(turbine_config, text='Solidity:')
sol_label.grid(row=7, column=0, padx=5, pady=5, sticky='e')
sol_entry = ttk.Entry(turbine_config, textvariable=sol_var)
sol_entry.grid(row=7, column=1, padx=5, pady=5)

omega_var = tk.StringVar(value='5.03')
omega_label = ttk.Label(turbine_config, text='Omega:')
omega_label.grid(row=8, column=0, padx=5, pady=5, sticky='e')
omega_entry = ttk.Entry(turbine_config, textvariable=omega_var)
omega_entry.grid(row=8, column=1, padx=5, pady=5)

ntheta_var = tk.StringVar(value='36')
ntheta_label = ttk.Label(turbine_config, text='Ntheta:')
ntheta_label.grid(row=9, column=0, padx=5, pady=5, sticky='e')
ntheta_entry = ttk.Entry(turbine_config, textvariable=ntheta_var)
ntheta_entry.grid(row=9, column=1, padx=5, pady=5)

vinf_var = tk.StringVar(value='5.03')
vinf_label = ttk.Label(turbine_config, text='Vinf Velocity:')
vinf_label.grid(row=10, column=0, padx=5, pady=5, sticky='e')
vinf_entry = ttk.Entry(turbine_config, textvariable=vinf_var)
vinf_entry.grid(row=10, column=1, padx=5, pady=5)

airfoil_var = tk.StringVar(value='naca0021')
airfoil_label = ttk.Label(turbine_config, text='Airfoil:')
airfoil_label.grid(row=11, column=0, padx=5, pady=5, sticky='e')
airfoil_entry = ttk.Entry(turbine_config, textvariable=airfoil_var)
airfoil_entry.grid(row=11, column=1, padx=5, pady=5)

# Radio frame buttons

fixed_param_var = tk.StringVar(value='omega')
omega_radio = ttk.Radiobutton(radio_buttons_frame, text='Fix Omega', variable=fixed_param_var, value='omega')
omega_radio.grid(row=1, column=0, padx=10)

vinf_radio = ttk.Radiobutton(radio_buttons_frame, text='Fix Vinf', variable=fixed_param_var, value='vinf')
vinf_radio.grid(row=1, column=1, padx=10)

#var_vel_var = tk.BooleanVar()
#var_vel_check = ttk.Checkbutton(radio_buttons_frame, text='Var Omega', variable=var_vel_var)
#var_vel_check.grid(row=1, column=0, padx=12, pady=10)

save_check_var = tk.BooleanVar()
save_check = ttk.Checkbutton(radio_buttons_frame, text='Export data (.dat / .csv) ', variable=save_check_var)
save_check.grid(row=1, column=2, padx=12, pady=10)

save_config_var = tk.BooleanVar()
save_config_check = ttk.Checkbutton(radio_buttons_frame, text='Save simulations settings', variable=save_config_var)
save_config_check.grid(row=1, column=3, padx=12, pady=10)

save_plot_var = tk.BooleanVar()
save_plot_check = ttk.Checkbutton(radio_buttons_frame, text='Save cp x tsr plot', variable=save_plot_var)
save_plot_check.grid(row=1, column=4, padx=12, pady=10)

run_sim_button = ttk.Button(
    bottom_frame,
    text='Run simulation',
    command=run_from_gui,
    width=30,
    padding=10,
    )
run_sim_button.pack()

params = r_var, H_var, twist_var, delta_var, chord_var, B_var, sol_var, omega_var, ntheta_var, vinf_var, airfoil_var

# run
window.mainloop()