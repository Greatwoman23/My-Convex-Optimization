## Task
This project focuses on implementing various algorithms and methods for convex optimization.
The goal is to provide solutions to problems such as finding roots of equations, optimizing functions, and solving linear programming problems using the simplex methods.

## Description
The problem of convex optimization was addressed by implementing Python algorithms for specific tasks. 
Here's a brief overview of the solutions:

Plotting a Function:
Implemented a function to plot a given function and its specified values using Matplotlib.

Finding Root using Bisection Method:
Developed a function to find the root of a function within a specified interval using the bisection method.

Finding Root using Newton-Raphson Method:
Implemented a function to find the root of a function using the Newton-Raphson method.

Gradient Descent Optimization:
Developed a function to perform gradient descent optimization on a function using its derivative.

Solving Linear Programming Problem using Simplex Method:
Implemented a function to solve a linear programming problem using the simplex method from the SciPy library.

These solutions utilize mathematical techniques and algorithms to efficiently solve specific convex optimization problems.

## Installation
The following libraries are installed:
NumPy
SciPy
Matplotlib

## Usage
The project consists of several functions and algorithms for convex optimization.
 Here's a brief description of each function:
print_a_function(f, values): This function plots a given function f(x) and displays a set of values on the plot.

find_root_bisection(f, a, b, precision=0.001): This function uses the bisection method to find the root of a given function f(x) within the interval [a, b].

find_root_newton_raphson(f, f_deriv, start, precision=0.001): This function applies Newton-Raphson method to find the root of a given function f(x) using its derivative f_deriv(x).

gradient_descent(f, f_prime, start, learning_rate=0.1, precision=0.001): This function performs gradient descent optimization on a given function f(x) using its derivative f_prime(x).

solve_linear_problem(A, b, c): This function solves a linear programming problem using the simplex method.
To use the functions, you can import the module and call the desired function with the appropriate arguments.

```
./ How to Run the Program
To run the program, follow these steps:
Run the program using the command python convex_optimization.py.
The program will execute and display the results and plots corresponding to the implemented functions.

```

### The Core Team


<span><i>Made at <a href='https://qwasar.io'>Qwasar SV -- Software Engineering School</a></i></span>
<span><img alt='Qwasar SV -- Software Engineering School's Logo' src='https://storage.googleapis.com/qwasar-public/qwasar-logo_50x50.png' width='20px'></span>
