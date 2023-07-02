from timeit import timeit

def f(x):
    return x ** 2 - x


def integrate_f(a, b, N):
    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += f(a + i * dx)
    return s * dx

def main():
    integrate_f(10, 100, 10000000)

print(f"total time : {timeit(main, number=1)}")