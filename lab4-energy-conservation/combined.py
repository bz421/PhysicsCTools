import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# ke v t
x_ke = np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])  # t
y_ke = np.array([980.88626, 770.49849, 620.22154, 530.05538, 500, 530.05538, 620.22154, 770.49849, 980.88626])  # ke

# pe v t
x_pe = np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])  # t
y_pe = np.array([0, 210.3877, 360.6647, 450.8308, 480.8862, 450.8308, 360.6647, 210.3877, 0])  # pe

# me v t
x_me = np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])  # t
y_me = np.array([980.88626, 980.8862, 980.8862, 980.8862, 980.8862, 980.8862, 980.8862, 980.8862, 980.88626])  # me

def smooth_curve(x, y, num_points=300):
    x_smooth = np.linspace(x.min(), x.max(), num_points)
    spline = make_interp_spline(x, y, k=3)  # Cubic spline
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth

# Curves
x_ke_smooth, y_ke_smooth = smooth_curve(x_ke, y_ke)
x_pe_smooth, y_pe_smooth = smooth_curve(x_pe, y_pe)
x_me_smooth, y_me_smooth = smooth_curve(x_me, y_me)

plt.figure(figsize=(10, 8), dpi=600)
plt.title('Energy vs. Time', fontsize=14)
plt.xlabel('Time ($s$)', fontsize=12)
plt.ylabel('Energy ($J$)', fontsize=12)

# plot ke
plt.plot(x_ke_smooth, y_ke_smooth, label='Kinetic Energy', color='blue')
plt.scatter(x_ke, y_ke, color='blue', s=10)

# plot pe
plt.plot(x_pe_smooth, y_pe_smooth, label='Gravitational Potential Energy', color='green')
plt.scatter(x_pe, y_pe, color='green', s=10)

# plot me
plt.plot(x_me_smooth, y_me_smooth, label='Mechanical Energy', color='red')
plt.scatter(x_me, y_me, color='red', s=10)

plt.legend(fontsize=10)
plt.grid(alpha=0.3)

plt.savefig('combined.png')
plt.show()