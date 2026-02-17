import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.odr import ODR, Model, RealData

# data
x = np.array([0.43, 0.884, 1.35, 1.7, 2.06, 2.39, 2.69, 2.94]) # a
y = np.array([0.05, 0.39, 0.77, 1.06, 1.46, 1.97, 2.26, 2.5])
y = np.array([0.05, 0.39, 0.77, 1.06, 1.46, 1.72, 1.95, 2.18]) # F
xerr = np.array([0.002, 0.005, 0.004, 0.01, 0.004, 0.01, 0.01, 0.01])
yerr = np.array([0.02, 0.03, 0.07, 0.07, 0.09, 0.06, 0.16, 0.37])

def linear(B, x):
    return B[0]*x + B[1]

model = Model(linear)

# Package data with uncertainties
data = RealData(x, y, sx=xerr, sy=yerr)

# ODR setup (initial guess: slope=1, intercept=0)
odr = ODR(data, model, beta0=[1., 0.])
out = odr.run()

# Best fit parameters
m, b = out.beta
dm, db = out.sd_beta  # 1-sigma uncertainties

print(f"Slope = {m:.3f} ± {dm:.3f}")
print(f"Intercept = {b:.3f} ± {db:.3f}")

print(m, m-dm, m+dm)

# Generate fit lines
xfit = np.linspace(min(x)-0.5, max(x)+0.5, 100)
yfit = linear([m, b], xfit)

# Min/max slope lines
yfit_min = linear([m - dm, b - db], xfit)
yfit_max = linear([m + dm, b + db], xfit)

# Plot
plt.figure(figsize=(8, 6), dpi=600)
plt.title('Force vs. Acceleration: Best-fit lines')
plt.xlabel('Acceleration ($m/s^2$)')
plt.ylabel('Force ($N$)')
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', markersize=4, label="Data", capsize=3)
plt.plot(xfit, yfit, 'r-', label="Best fit")
plt.plot(xfit, yfit_min, 'g--', label="Min slope")
plt.plot(xfit, yfit_max, 'b--', label="Max slope")
# plt.xlim(left=0)
# plt.ylim(bottom=0)
plt.legend()
plt.savefig('newton.png')
plt.show()
