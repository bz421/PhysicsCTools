import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.odr import ODR, Model, RealData

# data
x = np.array([10.0000005,
15.0000005,
20.0000005,
25.0000005,
30.0000005,
35.0000005,
40.0000005,
45.0000005,
50.0000005,
60.0000005
]) # k

y = np.array([2.00,
1.60,
1.40,
1.25,
1.15,
1.05,
1.00,
0.95,
0.90,
0.80
]) # T

xerr = np.array([0.0000005 for _ in range(len(x))])
yerr = np.array([0.05 for _ in range(len(y))])

def linear(B, x):
    return B[0] * x + B[1]

model = Model(linear)

log_x = np.log(x)
log_y = np.log(y)
log_xerr = xerr / x
log_yerr = yerr / y

log_data = RealData(log_x, log_y, sx=log_xerr, sy=log_yerr)
log_odr = ODR(log_data, model, beta0=[1., 0.])
log_out = log_odr.run()

log_m, log_b = log_out.beta
log_dm, log_db = log_out.sd_beta

print(f"Log Slope = {log_m:.6f} ± {log_dm:.15f}")
print(f"Log Intercept = {log_b:.6f} ± {log_db:.15f}")

def rotate_line(x_vals, y_vals, pivot_x, pivot_y, angle):
    x_rotated = pivot_x + (x_vals - pivot_x) * np.cos(angle) - (y_vals - pivot_y) * np.sin(angle)
    y_rotated = pivot_y + (x_vals - pivot_x) * np.sin(angle) + (y_vals - pivot_y) * np.cos(angle)
    return x_rotated, y_rotated

pivot_x = 30
pivot_y = linear([log_m, log_b], np.log(pivot_x))

# Generate fit lines
angle = np.radians(2)
log_xfit = np.linspace(min(log_x) - 0.5, max(log_x) + 0.5, 100)
log_yfit = linear([log_m, log_b], log_xfit)
log_xfit_min, log_yfit_min = rotate_line(log_xfit, log_yfit, np.log(pivot_x), pivot_y, -angle)
log_xfit_max, log_yfit_max = rotate_line(log_xfit, log_yfit, np.log(pivot_x), pivot_y, angle)

plt.figure(figsize=(8, 6), dpi=600)
plt.title('Period vs. Spring Constant ($m$ constant)')
plt.xlabel('Spring Constant ($N/m$)')
plt.ylabel('Period ($s$)')
plt.errorbar(log_x, log_y, xerr=log_xerr, yerr=log_yerr, fmt='o', markersize=4, label="Data", capsize=3)
plt.plot(log_xfit, log_yfit, 'r-', label="Best fit")
plt.plot(log_xfit, log_yfit_min, 'g--', label="Min slope")
plt.plot(log_xfit, log_yfit_max, 'b--', label="Max slope")

plt.legend()
plt.savefig('T-k.png')
plt.show()

while True:
    fit_type = input("Fit type: ").strip().lower()
    xin = float(input("x value: "))
    yout = -1

    if fit_type == "max":
        yout = linear([log_m + log_dm, log_b + log_db], xin)
    elif fit_type == "min":
        yout = linear([log_m - log_dm, log_b - log_db], xin)
    elif fit_type == "avg":
        yout = linear([log_m, log_b], xin)

    print(yout)
