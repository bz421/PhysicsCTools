import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.odr import ODR, Model, RealData

# data
x = np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]) # t
y = np.array([980.88626, 770.49849, 620.22154, 530.05538, 500, 530.05538, 620.22154, 770.49849, 980.88626]) # ke
xerr = np.array([0 for _ in range(len(x))])
yerr = np.array([0 for _ in range(len(y))])

def linear(B, x):
    return B[0]*x + B[1]

model = Model(linear)

data = RealData(x, y)

odr = ODR(data, model, beta0=[1., 0.])
out = odr.run()

m, b = out.beta

print(f'Slope = {m:.3f}')
print(f'Intercept = {b:.3f}')

xfit = np.linspace(min(x)-0.5, max(x)+0.5, 100)
yfit = linear([m, b], xfit)

plt.figure(figsize=(8, 6), dpi=600)
plt.title('Kinetic Energy vs. Time')
plt.xlabel('Time ($s$)')
plt.ylabel('Kinetic Energy ($J$)')
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', markersize=4, label="Data", capsize=3)
# plt.plot(xfit, yfit, 'r-', label="Best fit")
# plt.xlim(left=0)
# plt.ylim(bottom=0)
plt.legend()
plt.savefig('ke-t.png')
plt.show()
