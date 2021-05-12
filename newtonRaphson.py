def square(x):
    return x * x


def newton_raphson(f, precision=0.001, dx=0.001, max_iterations=1000):
    x0 = 1.0
    for i in range(1, max_iterations):
        y = f(x0)
        y_derivative = (f(x0 + dx) - f(x0 - dx)) / (2 * dx)
        if y_derivative == 0:
            return x0

        x1 = x0 - y / y_derivative
        if abs(x1 - x0) < precision:
            x0 = x1
            break

        x0 = x1

    return x0


# print(newton_raphson(square))
