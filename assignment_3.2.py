import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import chaospy as cp

# Transform the second order differential equation into a system of first order equations
def model(w, t, p):
    x1, x2 = w 
    c, k, f, omega = p  
    dx1dt = x2
    dx2dt = f * np.cos(omega * t) - k * x1 - c * x2
    return [dx1dt, dx2dt]

# Discretize the oscillator using the odeint function
def discretize_oscillator_odeint(model, init_cond, t, params, atol=1e-6, rtol=1e-6):
    sol = odeint(model, init_cond, t, args=(params,), atol=atol, rtol=rtol)
    return sol

if __name__ == '__main__':
    # Parameters
    c = 0.5   # damping coefficient
    k = 20.0  # spring constant
    f = 10.0  # forcing amplitude
    omega = 0.5  # forcing frequency
    y0 = 1.0  # initial position
    y1 = 0.5  # initial velocity

    # Time domain setup
    t = np.linspace(0, 20, 2001)  # time from 0 to 20 seconds

    # Initial conditions and parameters setup
    init_cond = [y0, y1]
    params = (c, k, f, omega)

    deterministic_sol = discretize_oscillator_odeint(model, init_cond, t, params)

    plt.figure(figsize=(10, 5))
    plt.plot(t, deterministic_sol[:, 0], label='Position (y)')
    plt.plot(t, deterministic_sol[:, 1], label='Velocity (dy/dt)')
    plt.title('Deterministic Solution of Damped Oscillator')
    plt.xlabel('Time (s)')
    plt.ylabel('Response')
    plt.legend()
    plt.show()
