import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.odr import ODR, Model, RealData

# data
x = np.array([
1.0000005,
2.0000005,
4.0000005,
5.0000005,
8.0000005,
10.0000005,
12.5000005,
20.0000005,
25.0000005,
40.0000005,
]) # r

y = np.array([
1.0000005,
0.5000005,
0.2500005,
0.2000005,
0.1250005,
0.1000005,
0.0800005,
0.0500005,
0.0400005,
0.0250005,
]) # a

xerr = np.array([0.0000005 for _ in range(len(x))])
yerr = np.array([0.0000005 for _ in range(len(y))])

def linear(B, x):
    return B[0]*x + B[1]

model = Model(linear)

log_x = np.log(x)
log_y = np.log(y)
print(log_x)
print(log_y)
log_xerr = xerr / x
log_yerr = yerr / y

log_data = RealData(log_x, log_y, sx=log_xerr, sy=log_yerr)
log_odr = ODR(log_data, model, beta0=[1., 0.])
log_out = log_odr.run()

log_m, log_b = log_out.beta
log_dm, log_db = log_out.sd_beta

print(f"Log Slope = {log_m:.6f} ± {log_dm:.15f}")
print(f"Log Intercept = {log_b:.6f} ± {log_db:.15f}")

log_xfit = np.linspace(min(log_x) - 0.5, max(log_x) + 0.5, 100)
log_yfit = linear([log_m, log_b], log_xfit)

print(linear([log_m, log_b], 1))

log_yfit_min = linear([log_m-log_dm, log_b-log_db], log_xfit)
log_yfit_max = linear([log_m+log_dm, log_b+log_db], log_xfit)

plt.figure(figsize=(8, 6), dpi=600)
plt.title('Log-Log Plot: Acceleration vs. Radius')
plt.xlabel('log(Radius $[m]$)')
plt.ylabel('log(Acceleration $[m/s^2]$)')
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.errorbar(log_x, log_y, xerr=log_xerr, yerr=log_yerr, fmt='o', markersize=4, label="Data", capsize=3)
plt.plot(log_xfit, log_yfit, 'r-', label="Best fit")
plt.plot(log_xfit, log_yfit_min, 'g--', label="Min slope")
plt.plot(log_xfit, log_yfit_max, 'b--', label="Max slope")
plt.legend()
plt.savefig('a-r.png')
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