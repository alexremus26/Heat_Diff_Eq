import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import time

# === Domain parameters ===
a = 5
b = 2
n_vals = [4,8,16,32,64,128,256]  # Different grid sizes to visualize
max_errors = []


# === Boundary functions ===
def este_in_GammaD(x, y):
    return np.isclose(y, 0) or np.isclose(y, b)


def este_in_GammaN(x, y):
    return np.isclose(x, 0) or np.isclose(x, a)


def gD(x, y):
    return u(x, y)


def gN(x, y, hx):
    k_val = k(x, y)
    if np.isclose(x, 0):
        du_dn = (u(x + hx, y) - u(x, y)) / hx
    elif np.isclose(x, a):
        du_dn = (u(x, y) - u(x - hx, y)) / hx
    else:
        raise ValueError("gN defined only for x=0 or x=a")
    return -k_val * du_dn


# === QR solver (preserved as requested) ===
def rezolva_sistem_QR(A, b):
    A = A.astype(float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] == 0:
            raise ValueError("Linearly dependent columns - QR failed")
        Q[:, j] = v / R[j, j]
    c = Q.T @ b
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (c[i].item() - np.dot(R[i, i + 1:], x[i + 1:])) / R[i, i].item()
    return x.reshape(-1, 1)


# === Exact functions and coefficients ===
def u(x, y):
    return np.sin(2 * np.pi * x / 3) * np.cos(np.pi * y / 2)


def f(x, y, hx, hy):
    # Central second-order derivatives
    du_dx = (u(x + hx, y) - u(x - hx, y)) / (2 * hx)
    du_dy = (u(x, y + hy) - u(x, y - hy)) / (2 * hy)

    dk_dx = (k(x + hx, y) - k(x - hx, y)) / (2 * hx)
    dk_dy = (k(x, y + hy) - k(x, y - hy)) / (2 * hy)

    d2u_dx2 = (u(x + hx, y) - 2 * u(x, y) + u(x - hx, y)) / hx ** 2
    d2u_dy2 = (u(x, y + hy) - 2 * u(x, y) + u(x, y - hy)) / hy ** 2

    term_x = dk_dx * du_dx + k(x, y) * d2u_dx2
    term_y = dk_dy * du_dy + k(x, y) * d2u_dy2

    return -(term_x + term_y)


def k(x, y):
    return 1 + 0.5 * np.sin(2 * np.pi * x) * np.exp(-y)


# === Build sparse system ===
def construieste_sistem_sparse(N_val):
    hx = a / N_val
    hy = b / N_val
    hx2 = hx ** 2
    hy2 = hy ** 2
    n_total = (N_val + 1) ** 2

    # Use LIL format for efficient construction
    A = lil_matrix((n_total, n_total))
    b_vec = np.zeros(n_total)

    node = lambda i, j: i + j * (N_val + 1)

    for j in range(N_val + 1):
        for i in range(N_val + 1):
            idx = node(i, j)
            x = i * hx
            y = j * hy

            if este_in_GammaD(x, y):
                A[idx, idx] = 1
                b_vec[idx] = gD(x, y)

            elif este_in_GammaN(x, y):
                if i == 0:  # Left boundary
                    k_avg = 0.5 * (k(x, y) + k(x + hx, y))
                    A[idx, idx] = -k_avg / hx
                    A[idx, node(i + 1, j)] = k_avg / hx
                    b_vec[idx] = gN(x, y, hx)

                elif i == N_val:  # Right boundary
                    k_avg = 0.5 * (k(x, y) + k(x - hx, y))
                    A[idx, idx] = k_avg / hx
                    A[idx, node(i - 1, j)] = -k_avg / hx
                    b_vec[idx] = gN(x, y, hx)

            else:  # Interior node
                k_c = k(x, y)
                A[idx, node(i, j - 1)] = -k_c / hy2
                A[idx, node(i - 1, j)] = -k_c / hx2
                A[idx, idx] = 2 * k_c * (1 / hx2 + 1 / hy2)
                A[idx, node(i + 1, j)] = -k_c / hx2
                A[idx, node(i, j + 1)] = -k_c / hy2
                b_vec[idx] = f(x, y, hx, hy)

    return csr_matrix(A), b_vec


# === Quadratic spline interpolation (optimized) ===
def spline_patratica_1d(X, Y):
    n = len(X) - 1
    A = np.zeros((3 * n, 3 * n))
    b = np.zeros(3 * n)
    row = 0

    # Point matching conditions
    for i in range(n):
        A[row, 3 * i:3 * i + 3] = [0, 0, 1]
        b[row] = Y[i]
        row += 1

        dx = X[i + 1] - X[i]
        A[row, 3 * i:3 * i + 3] = [dx ** 2, dx, 1]
        b[row] = Y[i + 1]
        row += 1

    # Continuity of derivatives
    for i in range(n - 1):
        dx = X[i + 1] - X[i]
        A[row, 3 * i:3 * i + 3] = [2 * dx, 1, 0]
        A[row, 3 * (i + 1) + 1] = -1
        row += 1

    # Natural spline condition
    A[row, 0] = 2
    row += 1

    # Solve system
    coef = rezolva_sistem_QR(A[:row], b[:row])

    def eval_spline(x):
        i = np.searchsorted(X, x) - 1
        i = max(0, min(i, n - 1))
        dx = x - X[i]
        a, b, c = coef[3 * i:3 * i + 3].flatten()
        return a * dx ** 2 + b * dx + c

    return eval_spline


def spline_bi2d(X, Y, Z):
    row_splines = []
    for j in range(len(Y)):
        row_splines.append(spline_patratica_1d(X, Z[j, :]))

    col_spline = spline_patratica_1d(Y, np.array([s(0.5) for s in row_splines]))

    def eval_bi2d(x, y):
        # Vectorized evaluation
        z_vals = np.array([spline(x) for spline in row_splines])
        col_spline = spline_patratica_1d(Y, z_vals)
        return col_spline(y)

    return eval_bi2d


# === Visualization function ===
def plot_solutions(X, Y, Z_num, Z_exact, N_val):
    error = np.abs(Z_num - Z_exact)
    max_error = np.max(error)
    print(f"N = {N_val}, Max error = {max_error:.6f}")

    fig = plt.figure(figsize=(18, 6))

    # Numerical solution
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, Z_num, cmap='viridis')
    ax1.set_title(f'Numerical Solution (N={N_val})')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # Exact solution
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y, Z_exact, cmap='plasma')
    ax2.set_title('Exact Solution')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    # Error
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X, Y, error, cmap='coolwarm')
    ax3.set_title('Absolute Error')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')

    plt.tight_layout()
    plt.show()

    return max_error


# === Main loop over grid sizes ===
for N_val in n_vals:
    print(f"\n=== Solving for N = {N_val} ===")
    start_time = time.time()

    # Build and solve system
    A_sparse, b_vec = construieste_sistem_sparse(N_val)
    U = spsolve(A_sparse, b_vec)

    # Prepare grid
    x_vals = np.linspace(0, a, N_val + 1)
    y_vals = np.linspace(0, b, N_val + 1)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z_vals = U.reshape((N_val + 1, N_val + 1))

    # Create spline interpolant
    spline2d = spline_bi2d(x_vals, y_vals, Z_vals)

    # Evaluate on dense grid
    x_dense = np.linspace(0, a, 50)
    y_dense = np.linspace(0, b, 50)
    X_dense, Y_dense = np.meshgrid(x_dense, y_dense)

    # Vectorized evaluation
    Z_dense = np.zeros_like(X_dense)
    for j in range(len(y_dense)):
        for i in range(len(x_dense)):
            Z_dense[j, i] = spline2d(X_dense[j, i], Y_dense[j, i])

    # Exact solution
    Z_exact = u(X_dense, Y_dense)

    # Plot and record error
    max_error = plot_solutions(X_dense, Y_dense, Z_dense, Z_exact, N_val)
    max_errors.append(max_error)

    print(f"Solved in {time.time() - start_time:.2f} seconds")

# === Convergence analysis ===
plt.figure(figsize=(10, 6))
plt.plot(n_vals, max_errors, 'o-', linewidth=2)
plt.xlabel('Number of intervals (N)')
plt.ylabel('Maximum Absolute Error')
plt.title('Convergence Analysis')
plt.grid(True, linestyle='--', alpha=0.7)
plt.yscale('log')
plt.xscale('log')
plt.show()
