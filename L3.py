#L3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def Lotka_Volterra_Euler():
    def dxdt(t, x, y, a, b, c, d):
        return (a - b * y) * x

    def dydt(t, x, y, a, b, c, d):
        return (c * x - d) * y

    def euler_metoda(a, b, c, d, x0, y0, t_start, t_end, dt):
        t_span = np.arange(t_start, t_end + dt, dt)
        x_result = np.zeros_like(t_span)
        y_result = np.zeros_like(t_span)
        x_result[0] = x0
        y_result[0] = y0

        for i in range(1, len(t_span)):
            t = t_span[i - 1]
            x = x_result[i - 1]
            y = y_result[i - 1]
            x_result[i] = x + dt * dxdt(t, x, y, a, b, c, d)
            y_result[i] = y + dt * dydt(t, x, y, a, b, c, d)

        return t_span, x_result, y_result

    a = 1.2
    b = 0.6
    c = 0.3
    d = 0.8

    x0 = 2
    y0 = 1

    t_start = 0.01
    t_end = 25
    dt = 0.01

    # Metoda Eulera
    t_span, x_result, y_result = euler_metoda(a, b, c, d, x0, y0, t_start, t_end, dt)

    # Wykres wyników
    plt.plot(t_span, x_result, label='x(t)')
    plt.plot(t_span, y_result, label='y(t)')
    plt.xlabel('Czas')
    plt.ylabel('Populacja')
    plt.title('Lotka-Volterra metoda Eulera')
    plt.legend()
    plt.grid(True)
    # plt.show()

    return x_result, y_result


def Lotka_Volterra_odeint():
    def lotka_volterra(y, t, a, b, c, d):
        x, y = y
        dxdt = (a - b * y) * x
        dydt = (c * x - d) * y
        return [dxdt, dydt]

    a = 1.2
    b = 0.6
    c = 0.3
    d = 0.8

    initial_conditions = [2, 1]

    t = np.arange(0, 25, 0.01)

    # Rozwiązanie
    odeint_solution = odeint(lotka_volterra, initial_conditions, t, args=(a, b, c, d))

    # Wykres
    plt.figure(figsize=(12, 6))
    plt.plot(t, odeint_solution[:, 0], label='x(t)')
    plt.plot(t, odeint_solution[:, 1], label='y(t)')
    plt.title("Lotka-Volterra odeint")
    plt.xlabel("Czas")
    plt.ylabel("Populacja")
    plt.legend()
    plt.grid(True)
    # plt.show()

    return odeint_solution[:, 0], odeint_solution[:, 1]


def Lorenz_Euler():
    def dxdt(t, x, y, z, sigma, ro, beta):
        return sigma * (y - x)

    def dydt(t, x, y, z, sigma, ro, beta):
        return x * (ro - z) - y

    def dzdt(t, x, y, z, sigma, ro, beta):
        return x * y - beta * z

    def euler_metoda(sigma, ro, beta, x0, y0, z0, t_start, t_end, dt):
        t_span = np.arange(t_start, t_end + dt, dt)
        x_result = np.zeros_like(t_span)
        y_result = np.zeros_like(t_span)
        z_result = np.zeros_like(t_span)
        x_result[0] = x0
        y_result[0] = y0
        z_result[0] = z0

        for i in range(1, len(t_span)):
            t = t_span[i - 1]
            x = x_result[i - 1]
            y = y_result[i - 1]
            z = z_result[i - 1]
            x_result[i] = x + dt * dxdt(t, x, y, z, sigma, ro, beta)
            y_result[i] = y + dt * dydt(t, x, y, z, sigma, ro, beta)
            z_result[i] = z + dt * dzdt(t, x, y, z, sigma, ro, beta)

        return t_span, x_result, y_result, z_result

    sigma = 10
    ro = 28
    beta = 8 / 3

    x0 = 1
    y0 = 1
    z0 = 1

    t_start = 0.002
    t_end = 25
    dt = 0.002

    # Metoda Eulera
    t_span, x_result, y_result, z_result = euler_metoda(sigma, ro, beta, x0, y0, z0, t_start, t_end, dt)

    # Wykres wyników
    # Wykres y(x)
    plt.figure(figsize=(10, 4))

    plt.subplot(2, 3, 1)
    plt.plot(x_result, y_result)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Lorenz: y(x)')

    # Wykres z(x)
    plt.subplot(2, 3, 2)
    plt.plot(x_result, z_result)
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Lorenz: z(x)')

    # Wykres z(y)
    plt.subplot(2, 3, 3)
    plt.plot(y_result, z_result)
    plt.xlabel('y')
    plt.ylabel('z')
    plt.title('Lorenz: z(y)')

    #plt.show()

    # Wykres 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x_result, y_result, z_result, lw=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Układ Lorenza')

    plt.show()
    return x_result, y_result, z_result


def Lorenz_odeint():
    def lorenz_system(y, t, sigma, ro, beta):
        x, y, z = y
        dxdt = sigma * (y - x)
        dydt = x * (ro - z) - y
        dzdt = x * y - beta * z
        return [dxdt, dydt, dzdt]

    sigma = 10
    ro = 28
    beta = 8 / 3

    initial_conditions = [1, 1, 1]

    t = np.arange(0, 25, 0.002)  # krok

    # Rozwiazanie
    odeint_solution = odeint(lorenz_system, initial_conditions, t, args=(sigma, ro, beta))

    # Wykres wyników
    # Wykres y(x)
    plt.figure(figsize=(10, 4))

    plt.subplot(2, 3, 1)
    plt.plot(odeint_solution[:, 0], odeint_solution[:, 1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Lorenz: y(x)')

    # Wykres z(x)
    plt.subplot(2, 3, 2)
    plt.plot(odeint_solution[:, 0], odeint_solution[:, 2])
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Lorenz: z(x)')

    # Wykres z(y)
    plt.subplot(2, 3, 3)
    plt.plot(odeint_solution[:, 1], odeint_solution[:, 2])
    plt.xlabel('y')
    plt.ylabel('z')
    plt.title('Lorenz: z(y)')

    # plt.show()

    # Wykres 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(odeint_solution[:, 0], odeint_solution[:, 1], odeint_solution[:, 2], lw=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trajektoria układu Lorenza')

    # plt.show()

    return odeint_solution[:, 0], odeint_solution[:, 1], odeint_solution[:, 2]


def Blad_Aproksymacyjny(euler, odeint):
    error = np.abs(euler - odeint)
    return np.mean(error)

#błąd aproksymacyjny dla lotki-volterry
euler_x, euler_y = Lotka_Volterra_Euler()
odeint_x, odeint_y = Lotka_Volterra_odeint()
print("blad aproksymacyjny Lotka-Volterra x:", Blad_Aproksymacyjny(euler_x, odeint_x))
print("blad aproksymacyjny Lotka-Volterra y:", Blad_Aproksymacyjny(euler_y, odeint_y))

#błąd aproksymacyjny dla lorenza
euler_x, euler_y, euler_z = Lorenz_Euler()
odeint_x, odeint_y, odeint_z = Lorenz_odeint()
print("blad aproksymacyjny Lorenz x:", Blad_Aproksymacyjny(euler_x, odeint_x))
print("blad aproksymacyjny Lorenz y:", Blad_Aproksymacyjny(euler_y, odeint_y))
print("blad aproksymacyjny Lorenz z:", Blad_Aproksymacyjny(euler_z, odeint_z))