import numpy as np
from mpmath import mp, mpf, log, linspace, fsum, sqrt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.odr import ODR, Model, RealData

dps = 100

x = [mpf(5.8e10), mpf(1.1e11), mpf(1.5e11), mpf(2.3e11)]
y = [mpf(7.6e6), mpf(19.8e6), mpf(31.6e6), mpf(59.95e6)]

log_x = [log(xi) for xi in x]
log_y = [log(yi) for yi in y]

xerr = [mpf(5e8), mpf(5e9), mpf(5e9), mpf(5e9)]
yerr = [mpf(25000) for _ in range(len(y))]

log_xerr = [dx / xi for dx, xi in zip(xerr, x)]
log_yerr = [dy / yi for dy, yi in zip(yerr, y)]

def linear(B, x):
    return B[0] * x + B[1]

w = [1 / (s ** 2) for s in log_yerr]

S = fsum(w)
Sx = fsum(w[i] * log_x[i] for i in range(len(x)))
Sy = fsum(w[i] * log_y[i] for i in range(len(x)))
Sxx = fsum(w[i] * log_x[i] ** 2 for i in range(len(x)))
Sxy = fsum(w[i] * log_x[i] * log_y[i] for i in range(len(x)))

Delta = S * Sxx - Sx ** 2

log_m = (S * Sxy - Sx * Sy) / Delta
log_b = (Sxx * Sy - Sx * Sxy) / Delta

log_dm = sqrt(S / Delta)
log_db = sqrt(Sxx / Delta)

print(f"Log Slope = {log_m} ± {log_dm}")
print(f"Log Intercept = {log_b} ± {log_db}")

# fit lines
log_xfit = [val for val in np.linspace(float(min(log_x)) - 0.5, float(max(log_x)) + 0.5, 100)]
log_yfit = [float(log_m * mpf(x) + log_b) for x in log_xfit]
log_yfit_min = [float((log_m - log_dm) * mpf(x) + (log_b - log_db)) for x in log_xfit]
log_yfit_max = [float((log_m + log_dm) * mpf(x) + (log_b + log_db)) for x in log_xfit]

print(log_xfit)
print(log_yfit)

plt.figure(figsize=(8, 6), dpi=600)
plt.errorbar(
    [float(v) for v in log_x],
    [float(v) for v in log_y],
    xerr=[float(v) for v in log_xerr],
    yerr=[float(v) for v in log_yerr],
    fmt='o',
    label='Data',
    capsize=3
)

plt.plot(log_xfit, log_yfit, 'r-', label='Best fit')
plt.plot(log_xfit, log_yfit_min, 'g--', label='Min slope')
plt.plot(log_xfit, log_yfit_max, 'b--', label='Max slope')

plt.xlabel(r'$\log(a)$')
plt.ylabel(r'$\log(T)$')
plt.title('Kepler’s Third Law (log–log)')
plt.legend()
plt.savefig('keplers-law-arbitrary.png')
plt.show()

while True:
    fit_type = input("Fit type: ").strip().lower()
    xin = mpf(input("x value: "))
    yout = None

    if fit_type == "max":
        yout = linear([log_m + log_dm, log_b + log_db], xin)
    elif fit_type == "min":
        yout = linear([log_m - log_dm, log_b - log_db], xin)
    elif fit_type == "avg":
        yout = linear([log_m, log_b], xin)
    else:
        print('Invalid')
        continue

    print(yout)

