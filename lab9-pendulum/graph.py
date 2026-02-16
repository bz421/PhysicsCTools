import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.odr import ODR, Model, RealData

# data
x = np.array([0.91300000,
0.80480000,
0.68450000,
0.63040000,
0.55670000,
0.48300000,
0.42580000,
0.37110000,
0.31090000,
0.25540000
])

y = np.array([1.92000000,
1.80000000,
1.67000000,
1.60000000,
1.50000000,
1.40000000,
1.29000000,
1.21000000,
1.12000000,
1.03000000])

xerr = np.array([0.00050249 for _ in x])
yerr = np.array([0.02 for _ in y])

y = y**2
yerr = 2 * yerr * y


def linear(B, x):
    return B[0]*x + B[1]

model = Model(linear)

data = RealData(x, y)

odr = ODR(data, model, beta0=[1., 0.])
out = odr.run()

m, b = out.beta
dm, db = out.sd_beta

print(f'Slope = {m:.3f} +/- {dm:.3f}')
print(f'Intercept = {b:.3f} +/- {db:.3f}')

xfit = np.linspace(min(x)-0.5, max(x)+0.5, 100)
yfit = linear([m, b], xfit)
yfit_min = linear([m - dm, b - db], xfit)
yfit_max = linear([m + dm, b + db], xfit)

plt.figure(figsize=(8, 6), dpi=600)
plt.title('Period squared vs. Length of a Simple Pendulum')
plt.xlabel('Length ($m$)')
plt.ylabel('Period ($s^2$)')
plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', markersize=4, label="Data", capsize=3)
plt.plot(xfit, yfit, 'r-', label="Best fit")
plt.plot(xfit, yfit_min, 'g--', label="Min slope")
plt.plot(xfit, yfit_max, 'b--', label="Max slope")

plt.legend()

plt.savefig('period-length.png')
plt.show()

while True:
    fit_type = input("Fit type: ").strip().lower()
    xin = float(input("x value: "))
    yout = None

    if fit_type == "max":
        yout = linear([m + dm, b + db], xin)
    elif fit_type == "min":
        yout = linear([m - dm, b - db], xin)
    elif fit_type == "avg":
        yout = linear([m, b], xin)
    else:
        print('Invalid')
        continue

    print(yout)