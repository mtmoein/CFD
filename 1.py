import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt

### Part1: Definition of governing equations

# f(Q,x) + S(x) = 0
# part1_1: Defining B.C. (i mean Q(0) & Q(1))
Q_l = 20.0  
Q_r = 50.0  

### Part2: Discretization of the domain

def solve_bvp(n, Q_l, Q_r):
    L = 1
    h = L / (n - 1)
    x_points = np.linspace(0, 1, n)

    ### Part3: Discretization of equations

    #f(Q,xi) + s(xi) = 0       # i = 2,3,...,N
    #after in code we define Q(0) & Q(1) with assuming values

    ### Part4: Converting the obtained equations into linear equations (making a sparse matrix)

    diagonals = [
        np.ones(n-1),             # 1 * u[i+1]
        -2 * np.ones(n),          # -2 * u[i]
        np.ones(n-1)              # 1 * u[i-1]
    ]
    offsets = [1, 0, -1]
    A = diags(diagonals, offsets, shape=(n, n)).toarray()

    # Inserting B.C. to matrix A
    A[0, :] = 0
    A[0, 0] = 1
    A[-1, :] = 0
    A[-1, -1] = 1

    # Define matrix b
    b = np.zeros(n)
    b[0] = Q_l
    b[-1] = Q_r
    b[1:-1] = h**2 * x_points[1:-1] * (1 - x_points[1:-1])

    ### Part5 : Solving the equations we obtain by gmres solution

    Q_interior, exitCode = gmres(A, b, atol=1e-5)
    if exitCode != 0:
        print("GMRES did not converge")

    # B.C. for Q
    Q = np.zeros(n)
    Q[0] = Q_l
    Q[1:-1] = Q_interior[1:-1]  # Adjust this line to skip the first and last elements of Q_interior
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
