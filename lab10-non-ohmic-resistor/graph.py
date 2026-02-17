import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData

# data
V = np.array([
0.03500000,
0.25500000,
0.43500000,
0.65500000,
0.84500000,
1.04500000,
1.21500000,
1.45500000,
1.64500000,
1.83500000,
2.03500000,
2.22500000,
2.44500000,
2.63500000,
2.83500000,
3.03500000])

R = np.array([
1.48619958,
1.92962543,
2.84592738,
3.87688665,
4.59363958,
5.09756098,
5.65116279,
6.19148936,
6.71428571,
6.92452830,
7.40000000,
7.80701754,
8.01639344,
8.36507937,
8.72307692,
9.05970149
])

Verr = np.array([0.005 for _ in V])
Rerr = np.array([
0.21233767,
0.03784284,
0.03272505,
0.02961679,
0.02720996,
0.12670052,
0.13346414,
0.13344097,
0.13853766,
0.13200684,
0.13576840,
0.13808424,
0.13243483,
0.13372444,
0.13508014,
0.13604065])

def resistance(B, x):
    a, b, R_0 = B
    return a * x ** 2 + b * x + R_0

model = Model(resistance)
data = RealData(V, R, sx=Verr, sy=Rerr)

odr = ODR(data, model, beta0=[-0.6, 4, 1])
output = odr.run()

a, b, R_0 = output.beta
da, db, dR_0 = output.sd_beta

print(f'a: {a} +/- {da}')
print(f'b: {b} +/- {db}')
print(f'R_0: {R_0} +/- {dR_0}')

x_fit = np.linspace(max(0, min(V) - 0.5), max(V) + 0.5, 100)
y_fit = resistance([a, b, R_0], x_fit)
y_new_fit = np.polyval(np.polyfit(V, R, 2), x_fit)


plt.figure(figsize=(8, 6), dpi=600)
plt.title('Resistance vs. Voltage')
plt.xlabel('Voltage ($V$)')
plt.ylabel('Resistance ($\Omega$)')
plt.errorbar(V, R, xerr=Verr, yerr=Rerr, fmt='o', markersize=4, label='Data', capsize=3)
plt.plot(x_fit, y_fit, 'r-', label='Best fit')
# plt.plot(x_fit, y_new_fit, 'g--', label='Best new fit')
plt.legend()
plt.savefig('res.png')

# y_fit = resistance([a, b, R_0], V)

alpha = 0.0045
temp = (R / R_0 - 1) / alpha + 24.0

coeff_1 = 1 / (R_0 * alpha) * Rerr
coeff_2 = -R / (R_0 ** 2 * alpha) * dR_0
coeff_3 = (R_0 - R) / (R_0 * alpha ** 2) * 0.0001
coeff_4 = 0.5

temps_from_v = (R / R_0 - 1) / alpha + 24.0
temp_err = np.sqrt(coeff_1**2 + coeff_2**2 + coeff_3**2 + coeff_4**2)
temp_err[0] = 53.01864048630587
print(temp_err)


def power(B, x):
    A, B, C = B
    return A * (x ** B) + C


temp_model = Model(power)
temp_data = RealData(V, temps_from_v, sx=Verr, sy=temp_err)

odr_temp = ODR(temp_data, temp_model, beta0=[764, 0.75, 24.0], ifixb=[1, 1, 0])
temp_output = odr_temp.run()

A, B, C = temp_output.beta
dA, dB, dC = temp_output.sd_beta
dC = 0.5

print(f'\nA: {A} +/- {dA}')
print(f'B: {B} +/- {dB}')
print(f'C: {C} +/- {dC}')

V_smooth = np.linspace(max(0, min(V) - 0.5), max(V) + 0.5, 100)
temp_smooth = power([A, B, C], V_smooth)

plt.figure(figsize=(8, 6), dpi=600)
plt.title('Temperature vs. Voltage')
plt.xlabel('Voltage ($V$)')
plt.ylabel('Temperature ($^\circ C$)')
plt.errorbar(V, temps_from_v, xerr=Verr, yerr=temp_err, fmt='o', markersize=4, label='Data', capsize=3)
plt.plot(V_smooth, temp_smooth, 'r-', label='Best fit')
# plt.plot(x_fit, y_new_fit, 'g--', label='Best new fit')
plt.legend()
plt.savefig('temp.png')

V_blow = ((3422 - C) / A) ** (1 / B)
dV_blow = V_blow * np.sqrt((dA / (A * B)) ** 2 + ((np.log((3422-C)/A) * dB)/(B ** 2)) ** 2 + (dC / (B * (3422 - C))) ** 2)

R_blow = resistance([a, b, R_0], V_blow)
dR_blow = np.sqrt((da * V_blow ** 2) ** 2 + (db * V_blow) ** 2 + dR_0 ** 2 + ((2 * a * V_blow + b) * dV_blow)**2)

I_blow = V_blow / resistance([a, b, R_0], V_blow)
dI_blow = I_blow * np.sqrt((dV_blow / V_blow) ** 2 + (dR_blow / R_blow) ** 2)

print(f'\nV_blow: {V_blow} +/- {dV_blow}')
print(f'R_blow: {R_blow} +/- {dR_blow}')
print(f'I_blow: {I_blow} +/- {dI_blow}')

while True:
    try:
        xin = float(input('Enter Custom Voltage: '))
        xout = resistance([a, b, R_0], xin)
        print(f'Resistance: {xout} +/- {np.sqrt((da * xin ** 2) ** 2 + (db * xin) ** 2 + dR_0 ** 2)}')
        print(f'Temp: {power([A, B, C], xin)} +/- {np.sqrt((dA * xin ** B) ** 2 + (dB * A * xin ** B * np.log(xin)) ** 2 + 0.5 ** 2)}')
    except Exception as e:
        print('Invalid input')

