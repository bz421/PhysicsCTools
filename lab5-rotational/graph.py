import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.odr import ODR, Model, RealData

# data
x = np.array([0.115000,
0.125000,
0.135000,
0.145000,
0.155000,
0.165000,
0.175000,
0.185000,
0.195000,
0.205000,
0.215000,
0.225000,
0.235000,
0.245000])

y = np.array([0.390965,
0.506372,
0.632831,
0.754287,
0.906827,
1.100238,
1.394110,
1.730827,
2.090959,
2.496679,
2.909294,
3.431272,
4.164823,
4.843177])

xerr = np.array([0.0005 for _ in range(len(x))])
yerr = np.array([0.030280,
0.038251,
0.047060,
0.055663,
0.065962,
0.078436,
0.096426,
0.117146,
0.138756,
0.162728,
0.187270,
0.217506,
0.259457,
0.297924])

x = x**2
xerr = 2*x*xerr

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
yfit_min = linear([m - dm, b - db], xfit)
yfit_max = linear([m + dm, b + db], xfit)

plt.figure(figsize=(8, 6), dpi=600)
plt.title('I vs. $R^2$')
plt.xlabel('$R^2$ ($m^2$)')
plt.ylabel('I ($kg \cdot m^2$)')
plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', markersize=4, label="Data", capsize=3)
plt.plot(xfit, yfit, 'r-', label="Best fit")
plt.plot(xfit, yfit_min, 'g--', label="Min slope")
plt.plot(xfit, yfit_max, 'b--', label="Max slope")
plt.legend()

plt.xlim(0.01, 0.061)
plt.ylim(0, 6)

plt.savefig('I-R^2.png')
plt.show()