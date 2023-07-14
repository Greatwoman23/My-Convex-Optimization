import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar


# Function f(x) = (x - 1)^4 + x^2
f = lambda x: (x - 1) ** 4 + x ** 2

def print_a_function(f, values):
    x = np.linspace(min(values), max(values), 100)
    y = f(x)
    plt.plot(x, y, color='blue', label='f(x)')
    plt.scatter(values, f(values), color='red', marker='x', label='Values')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Plot of the function f(x) with Values')
    plt.grid(True)
    plt.legend()
    plt.show()

def find_root_bisection(f, a, b, precision=0.001):
    while abs(b - a) > precision:
        c = (a + b) / 2
        if f(c) == 0:
            return c
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2

def find_root_newton_raphson(f, f_deriv, start, precision=0.001):
    x = start
    while abs(f(x)) > precision:
        x = x - f(x) / f_deriv(x)
    return x

def gradient_descent(f, f_prime, start, learning_rate=0.1, precision=0.001):
    x = start
    while abs(f_prime(x)) > precision:
        x -= learning_rate * f_prime(x)
    return x

# Solve linear problem using Simplex method
def solve_linear_problem(A, b, c):
    from scipy.optimize import linprog

    res = linprog(c, A_ub=A, b_ub=b, method='simplex')

    return res.fun, res.x

# Plot the function f(x)
x = np.linspace(-2, 3, 100)
y = f(x)
plt.plot(x, y, color='blue', label='f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Plot of the function f(x)')
plt.grid(True)
plt.legend()
plt.show()

# Define the derivative of f(x)
f_prime = lambda x: 4 * ((x - 1) ** 3) + 2 * x

# Find the root of f_prime using the bisection method
root = find_root_bisection(f_prime, -2, 3)
print('Root of f\':', root)

# Use Brent's method for optimization to find the minimum of f
res = minimize_scalar(f, method='brent')
x_min = res.x
f_min = res.fun
print('x_min:', '{:.2f}'.format(x_min), 'f(x_min):', '{:.2f}'.format(f_min))

# Plot the function with the minimum
x = np.linspace(x_min - 1, x_min + 1, 100)
y = f(x)
plt.plot(x, y, color='blue', label='f(x)')
plt.scatter(x_min, f_min, color='red', marker='x', label='Minimum')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Plot of the function f(x) with Minimum')
plt.grid(True)
plt.legend()
plt.show()

# Gradient Descent method
start_point = -1
x_min_gradient = gradient_descent(f, f_prime, start_point, learning_rate=0.01)
f_min_gradient = f(x_min_gradient)
print('x_min_gradient:', '{:.2f}'.format(x_min_gradient), 'f(x_min_gradient):', '{:.2f}'.format(f_min_gradient))

# Plot the function with the gradient descent minimum
x = np.linspace(x_min_gradient - 1, x_min_gradient + 1, 100)
y = f(x)
plt.plot(x, y, color='blue', label='f(x)')
plt.scatter(x_min_gradient, f_min_gradient, color='red', marker='x', label='Minimum (Gradient Descent)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Plot of the function f(x) with Gradient Descent Minimum')
plt.grid(True)
plt.legend()
plt.show()

# Linear problem with Simplex method
A = np.array([[2, 1], [-4, 5], [1, -2]])
b = np.array([10, 8, 3])
c = np.array([-1, -2])

optimal_value, optimal_arg = solve_linear_problem(A, b, c)

print("The optimal value is:", optimal_value, "and is reached for x =", optimal_arg)