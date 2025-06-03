import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#dokładne rozwiązanie
g, L, t, theta0, omega0 = sp.symbols('g L t theta0 omega0')
theta = sp.Function('theta')(t)
ode = sp.Eq(theta.diff(t, 2) + (g / L) * theta, 0)

warunki_poczatkowe = {theta.subs(t, 0): theta0, theta.diff(t).subs(t, 0): omega0}
rozw_ogolne = sp.dsolve(ode, theta, ics=warunki_poczatkowe)
theta_rozw = rozw_ogolne.rhs

theta_an = sp.lambdify(t, theta_rozw.subs({theta0: 0.3, omega0: 0, g: 9.81, L: 1}), 'numpy')

# Numeryczne rozwiązanie
theta0 = 0.3
omega0 = 0
g = 9.81
L = 1

T = 10
dt = 0.01
warunki_poczatkowe_num = [theta0, omega0]

def rownanie_wahadla(t, y):
    theta, omega = y
    dydt = [omega,  - (g/L)*theta]
    return dydt

t_zakres = (0, T)
t_ocena = np.arange(0, T, dt)
rozw_num = solve_ivp(rownanie_wahadla, t_zakres, warunki_poczatkowe_num, t_eval=t_ocena)

# Wyniki
theta_numeryczne = rozw_num.y[0]
czas = rozw_num.t

theta_sympy = theta_an(czas)

mae = np.mean(np.abs(theta_sympy - theta_numeryczne))
mse = np.mean((theta_sympy - theta_numeryczne)**2)
print(mae, mse)

plt.figure(figsize=(10, 6))
plt.plot(czas, theta_numeryczne, label='Rozwiązanie Numeryczne', linestyle = "-")
plt.plot(czas, theta_sympy, label='Rozwiązanie Analityczne', linestyle = "--")
plt.xlabel('Czas [s]')
plt.ylabel('Theta [rad]')
plt.legend()
plt.grid(True)
plt.subplots_adjust(bottom=0.22, top=0.9)
plt.figtext(0.5, 0.05, f"Średni błąd bezwzględny = {mae}, Średni błąd kwadratowy = {mse}", ha="center", fontsize=12)
plt.show()

blad_bezwz = np.abs(theta_sympy - theta_numeryczne)
blad_kw = (theta_sympy - theta_numeryczne)**2

plt.figure(figsize=(10, 6))
ax = plt.subplot(1, 2, 1)
ax.plot(czas, blad_bezwz, label='Błąd bezwzględny', linestyle = "-")
plt.xlabel('Czas [s]')
plt.ylabel('Odchylenie od wyniku dokładnego')
plt.legend()
plt.grid(True)

ax1 = plt.subplot(1, 2, 2)
ax1.plot(czas, blad_kw, label='Błąd kwadratowy', linestyle = "--")
plt.xlabel('Czas [s]')
plt.ylabel('Odchylenie od wyniku dokładnego')
plt.legend()
plt.grid(True)
plt.subplots_adjust(bottom=0.22, top=0.9)
plt.figtext(0.5, 0.05, f"Średni błąd bezwzględny = {mae}, Średni błąd kwadratowy = {mse}", ha="center", fontsize=12)
plt.show()
