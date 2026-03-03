import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.odr import ODR, Model, RealData

I = np.array([
0.185,
0.135,
0.105,
0.085,
0.075,
0.065,
0.055,
0.045,
0.03705,
0.03365,
0.03185,
0.03065,
0.02905
])

V = np.array([
1.4845,
1.5055,
1.5135,
1.5265,
1.5305,
1.5365,
1.5395,
1.5415,
1.5465,
1.5495,
1.5505,
1.5515,
1.5535
])

dI = np.array([
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.00005,
0.00005,
0.00005,
0.00005,
0.00005
])

dV = np.array([0.0005 for _ in V])

def linear(B, x):
    return B[0]*x + B[1]

model = Model(linear)
data = RealData(I, V, sx=dI, sy=dV)

odr = ODR(data, model, beta0=[-0.4, 1.5])
out = odr.run()

m, b = out.beta
dm, db = out.sd_beta

print(f'Slope: {m} +/- {dm}')
print(f'Intercept: {b} +/- {db}')

I_fit = np.linspace(min(I)-0.1, max(I)+0.1, 100)

V_fit = linear([m, b], I_fit)
V_fit_min = linear([m - dm, b - db], I_fit)
V_fit_max = linear([m + dm, b + db], I_fit)

plt.figure(figsize=(8, 6), dpi=600)
plt.title('Voltage vs. Current')
plt.xlabel('Current (A)')
plt.ylabel('Voltage (V)')
plt.errorbar(I, V, xerr=dI, yerr=dV, fmt='o', markersize=4, label="Data", capsize=3)
plt.plot(I_fit, V_fit, 'r-', label="Best fit")
plt.plot(I_fit, V_fit_min, '--', color='purple', label="Min slope")
plt.plot(I_fit, V_fit_max, 'y--', label="Max slope")
plt.legend()

plt.savefig('voltage-current.png')
plt.show()

while True:
    fit_type = input("Fit type: ").strip().lower()
    I_in = float(input("Current (A): "))
    V_out = None

    if fit_type == "max":
        V_out = linear([m + dm, b + db], I_in)
    elif fit_type == "min":
        V_out = linear([m - dm, b - db], I_in)
    elif fit_type == "avg":
        V_out = linear([m, b], I_in)
    else:
        print('Invalid')
        continue

    print(f'Voltage: {V_out} V')

"""
Slope: -0.4540221248346111 +/- 0.01894826648423807
Intercept: 1.5650179327452647 +/- 0.0007847576699320178

Avg slope:
(0 A, 1.5650179327452647 V)
(0.10 A, 1.5196157202618035 V)

Min slope: 
(0 A, 1.5642331750753327 V)
(0.10 A, 1.5169361359434477 V)

Max slope:
(0 A, 1.5658026904151967 V)
(0.10 A, 1.5222953045801595 V)
"""