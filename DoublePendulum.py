import matplotlib.pyplot as plt
import numpy as np

import dataclasses
import functools

import numpy as np
from scipy import integrate

GRAVITATIONAL_ACCELERATION = 9.81

@dataclasses.dataclass(frozen=True)
class State:
  theta1: float | np.ndarray
  theta1_dot: float | np.ndarray
  theta2: float | np.ndarray
  theta2_dot: float | np.ndarray
  t: float | np.ndarray = 0.

def _double_pendulum_ode(
    unused_t: float, y: np.ndarray, L1: float, L2: float, m1: float, m2: float):
  """Internal function called by integrate.solve_ivp."""
  g = GRAVITATIONAL_ACCELERATION
  theta1, theta1_dot, theta2, theta2_dot = y

  delta_theta = theta1 - theta2
  sin_delta = np.sin(delta_theta)
  cos_delta = np.cos(delta_theta)

  M = np.array([[(m1+m2)*L1, m2*L2*cos_delta],
                [m2*L1*cos_delta, m2*L2]])
  rhs = np.array([
      -m2 * L2 * theta2_dot**2 * sin_delta - (m1+m2) * g * np.sin(theta1),
      m2 * L1 * theta1_dot**2 * sin_delta - m2 * g * np.sin(theta2)
  ])
  theta_double_dot = np.linalg.solve(M, rhs)
  theta1_double_dot, theta2_double_dot = theta_double_dot

  return [theta1_dot, theta1_double_dot, theta2_dot, theta2_double_dot]

def solve(
    duration: float,
    initial_state: State,
    L1: float,
    L2: float,
    m1: float,
    m2: float,
    num_eval_points: int = 10000) -> State:

  y0 = [initial_state.theta1, initial_state.theta1_dot, initial_state.theta2, initial_state.theta2_dot]
  t_eval = np.linspace(0, duration, num_eval_points)
  sol = integrate.solve_ivp(
    functools.partial(_double_pendulum_ode, L1=L1, L2=L2, m1=m1, m2=m2),
    [0, duration], y0, t_eval=t_eval, method='RK45')

  return State(
    t=sol.t,
    theta1=sol.y[0],
    theta1_dot=sol.y[1],
    theta2=sol.y[2],
    theta2_dot=sol.y[3]
  )

L1 = L2 = m1 = m2 = 1

# Initial conditions
initial_state = State(
    theta1 = np.pi, 
    theta1_dot = 0.0,
    theta2 = np.pi,
    theta2_dot = 0.0,
)

duration = 20

solution = solve(duration, initial_state, L1, L2, m1, m2)

# Plot the results
plt.figure(figsize=(10, 5))

plt.plot(solution.t, solution.theta1, label=r'$\theta_1$ (Pendulum 1)')
plt.plot(solution.t, solution.theta2, label=r'$\theta_2$ (Pendulum 2)')

plt.show()

