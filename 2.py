import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

### Part1: Definition of governing equations

# f(Q,x) + S(x) = 0
# part1_1: Defining B.C.
Q_l = 20.0  # Boundary condition at x = 0
Q_r = 50.0  # Boundary condition at x = 1

### Part2: Discretization of the domain

def solve_bvp(n, Q_l, Q_r):
    L = 1
    h = L / (n - 1)
    x_points = np.linspace(0, 1, n)

    ### Part3: Discretization of equations

    #f(Q,xi) + s(xi) = 0       # i = 2,3,...,N
    #after in code we define Q(0) & Q(1) with assuming values

    ### Part4: Converting the obtained equations into linear equations (making a sparse matrix)

    def central_difference_order2_error2(n, h):
        coeffs = [-1, 2, -1]
        diagonals = [
            np.full(n, coeffs[0]),  # 1 * u[i-1]
            np.full(n, coeffs[1]),  # -2 * u[i]
            np.full(n, coeffs[2])   # 1 * u[i+1]
        ]
        offsets = [-1, 0, 1]
        D = diags(diagonals, offsets, shape=(n, n))
        return D

    A = central_difference_order2_error2(n, h)

    # create answer matrix
    b = np.zeros(n)
    for i in range(n):
        x_i = i * h
        b[i] = h**2 * x_i * (1 - x_i)
    b[0] = Q_l
    b[-1] = Q_r
    #*** check point_2 (without boundary):***
    #print (b)

    # Print matrix A and vector b for debugging
    print("Matrix A:\n", A.toarray())
    print("Vector b:\n", b)
    
    # Solve the linear system
    Q_interior = spsolve(A, b)
    Q = np.zeros(n)
    Q[0] = Q_l
    Q[1:-1] = Q_interior[1:-1]
    Q[-1] = Q_r
    
    return x_points, Q

# Grid sizes
grid_sizes = [11, 21, 41, 81, 161, 321]
solutions = []

# Solve for each grid size
for n in grid_sizes:
    x, Q = solve_bvp(n, Q_l, Q_r)
    solutions.append((x, Q))

### Part6: Result with plotting

plt.figure(figsize=(10, 6))
for i, (x, Q) in enumerate(solutions):
    plt.plot(x, Q, marker='o', linestyle='-', label=f'n={grid_sizes[i]}')
plt.xlabel('x')
plt.ylabel('Q')
plt.title('Solution of the second-order differential equation')
plt.legend()
plt.grid(True)
plt.show()

### Part7: Error calculation and plotting

errors = []
for i in range(1, len(solutions)):
    x_fine, Q_fine = solutions[i]
    x_coarse, Q_coarse = solutions[i-1]
    Q_coarse_interp = np.interp(x_fine, x_coarse, Q_coarse)
    error = np.abs(Q_fine - Q_coarse_interp)
    errors.append((x_fine, error))

plt.figure(figsize=(10, 6))
for i, (x, error) in enumerate(errors):
    plt.plot(x, error, marker='o', linestyle='-', label=f'Error between n={grid_sizes[i+1]} and n={grid_sizes[i]}')
plt.xlabel('x')
plt.ylabel('Error')
plt.title('Error between successive grid sizes')
plt.legend()
plt.grid(True)
plt.show()
