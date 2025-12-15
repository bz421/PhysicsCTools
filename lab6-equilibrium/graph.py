import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.odr import ODR, Model, RealData

# data
x = np.array([10, 20, 30, 40, 50, 60, 70, 80])

y = np.array([0.49315822,
0.53275830,
0.58004805,
0.63840916,
0.77636741,
0.98064109,
1.42630862,
2.83627241])

xerr = np.array([0 for _ in range(len(x))])
yerr = np.array([0.00004967,
0.00005175,
0.00005433,
0.00005764,
0.00006586,
0.00007877,
0.00010857,
0.00020752])

def linear(B, x):
    return B[0]*x + B[1]

model = Model(linear)

data = RealData(x, y)

odr = ODR(data, model, beta0=[1., 0.])
out = odr.run()

m, b = out.beta
dm, db = out.sd_beta

print(f'Slope = {m:.3f}')
print(f'Intercept = {b:.3f}')

xfit = np.linspace(min(x)-0.5, max(x)+0.5, 100)
yfit = linear([m, b], xfit)
# yfit_min = linear([m - dm, b - db], xfit)
# yfit_max = linear([m + dm, b + db], xfit)

plt.figure(figsize=(8, 6), dpi=600)
plt.title('Average Tension vs. Angle from $180^\circ$')
plt.xlabel('Angle from $180^\circ$ (degrees)')
plt.ylabel('Average Tension ($N$)')
plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', markersize=4, label="Data", capsize=3)
plt.legend()

plt.savefig('angle-tension.png')
plt.show()