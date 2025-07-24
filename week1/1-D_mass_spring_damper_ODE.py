import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# System parameters
m = 1.0   # kg
k = 20.0  # N/m
c = 2.0   # Ns/m

# External force function (optional)
def external_force(t):
    return 0  # Change this to np.sin(t) for forced system

# ODE function: x'' = (1/m)*(F - c*x' - k*x)
def mass_spring_damper(t, y):
    x, x_dot = y
    dxdt = x_dot
    dx_dotdt = (1/m) * (external_force(t) - c*x_dot - k*x)
    return [dxdt, dx_dotdt]

# Initial conditions: [x0, v0]
y0 = [1.0, 0.0]  # Start with displacement=1 m, velocity=0

# Time span
t_span = (0, 10)                # seconds
t_eval = np.linspace(*t_span, 1000)  # evaluation points

# Solve ODE
solution = solve_ivp(mass_spring_damper, t_span, y0, t_eval=t_eval)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(solution.t, solution.y[0], label='Displacement x(t)')
plt.plot(solution.t, solution.y[1], label='Velocity ẋ(t)', linestyle='--')
plt.xlabel('Time [s]')
plt.ylabel('Response')
plt.title('1D Mass-Spring-Damper System')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
