import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.odr import ODR, Model, RealData

# data
x = np.array([5.8e10, 1.1e11, 1.5e11, 2.3e11]) # a
y = np.array([7.6e6, 19.8e6, 31.6e6, 59.95e6]) # T

xerr = np.array([5e8, 5e9, 5e9, 5e9])
yerr = np.array([25000 for _ in range(len(y))])

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

# Generate fit lines
log_xfit = [val for val in np.linspace(min(log_x) - 0.5, max(log_x) + 0.5, 100)]
log_yfit = [linear([log_m, log_b], val) for val in log_xfit]
log_yfit_min = [linear([log_m - log_dm, log_b - log_db], val) for val in log_xfit]
log_yfit_max = [linear([log_m + log_dm, log_b + log_db], val) for val in log_xfit]

print(log_xfit)
print(log_yfit)

plt.figure(figsize=(8, 6), dpi=600)
plt.title('Period vs. Average Distance from Sun')
plt.xlabel('Average Distance from Sun ($m$)')
plt.ylabel('Period ($s$)')
plt.errorbar(log_x, log_y, xerr=log_xerr, yerr=log_yerr, fmt='o', markersize=4, label="Data", capsize=3)
plt.plot(log_xfit, log_yfit, 'r-', label="Best fit")
plt.plot(log_xfit, log_yfit_min, 'g--', label="Min slope")
plt.plot(log_xfit, log_yfit_max, 'b--', label="Max slope")

plt.legend()
plt.savefig('keplers-law.png')
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
