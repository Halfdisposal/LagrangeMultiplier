from scipy.linalg import solve
import numpy as np
import sympy as sm
import plotly.graph_objects as go
import matplotlib.pyplot as plt  

# -------------------------------
#Define the independent variablrs
x, y, lam = sm.symbols('x y lam')

a = 0.5
b = 1
f = x + y
g = (x**2 / a) + (y**2 / b) - 1

#Find the PDEs of f, g

f_x = sm.diff(f, x)
f_y = sm.diff(f, y)
g_x = sm.diff(g, x)
g_y = sm.diff(g, y)


F1 = f_x - lam * g_x
F2 = f_y - lam * g_y
F3 = g

# Build the system vector and its Jacobian
M = sm.Matrix([F1, F2, F3])
J = M.jacobian([x, y, lam])

##Convert expressions to Functions
F_num = sm.lambdify((x, y, lam), M, 'numpy')
J_num = sm.lambdify((x, y, lam), J, 'numpy')


f_eval = sm.lambdify((x, y), f, 'numpy')
g_eval = sm.lambdify((x, y), g, 'numpy')


print(f"df/dx = {f_x}, df/dy = {f_y}")
print(f"dg/dx = {g_x}, dg/dy = {g_y}")

# Set an initial guess for [x, y, lam]
x0, y0 = -0.77, 0.67
lam0 = 1 / (2 * x0)
X = np.array([x0, y0, lam0], dtype=float)

tol = 1e-8
max_iter = 100
alpha = 1.0 

for i in range(max_iter):
    F_val = np.array(F_num(X[0], X[1], X[2]), dtype=float).flatten()
    J_val = np.array(J_num(X[0], X[1], X[2]), dtype=float)
    
    delta = solve(J_val, F_val)
    X_new = X - alpha * delta
    F_new = np.array(F_num(X_new[0], X_new[1], X_new[2]), dtype=float).flatten()
    
    while np.linalg.norm(F_new) > np.linalg.norm(F_val) and alpha > 1e-4:
        alpha *= 0.5
        X_new = X - alpha * delta
        F_new = np.array(F_num(X_new[0], X_new[1], X_new[2]), dtype=float).flatten()
        
    if np.linalg.norm(X_new - X) < tol:
        X = X_new
        break
        
    X = X_new
    alpha = 1.0

print(f"Converged in {i+1} iterations")
print(f"Solution: x = {X[0]}, y = {X[1]}, lam = {X[2]}")
sol_x, sol_y, sol_lam = X[0], X[1], X[2]
sol_f = f_eval(sol_x, sol_y)

# -------------------------------
#Ploting

x_range = np.linspace(-1.5, 1.5, 100)
y_range = np.linspace(-1.5, 1.5, 100)
X_grid, Y_grid = np.meshgrid(x_range, y_range)
Z_grid = f_eval(X_grid, Y_grid)
G_grid = g_eval(X_grid, Y_grid)

fig_temp, ax_temp = plt.subplots()
contour_obj = ax_temp.contour(X_grid, Y_grid, G_grid, levels=[0])
plt.close(fig_temp)

contour_segments = contour_obj.allsegs[0]
contour_traces = []
for seg in contour_segments:
    x_contour = seg[:, 0]
    y_contour = seg[:, 1]
    z_contour = f_eval(x_contour, y_contour)
    trace = go.Scatter3d(
        x=x_contour,
        y=y_contour,
        z=z_contour,
        mode='lines',
        line=dict(color='red', width=4),
        name='Constraint: g(x,y)=0'
    )
    contour_traces.append(trace)

surface = go.Surface(
    x=X_grid,
    y=Y_grid,
    z=Z_grid,
    opacity=0.6,
    colorscale='Viridis',
    colorbar=dict(title='f(x,y)')
)

solution_point = go.Scatter3d(
    x=[sol_x],
    y=[sol_y],
    z=[sol_f],
    mode='markers',
    marker=dict(size=8, color='black'),
    name='Computed Solution'
)

fig = go.Figure(data=[surface, solution_point] + contour_traces)
fig.update_layout(
    title='3D Visualization of f(x,y) with Constraint g(x,y)=0',
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='f(x,y)'
    )
)

fig.show()
