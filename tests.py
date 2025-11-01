from piyavsky_method import PiyavskyMethod


test_1 = ("f(x)=2*x+cos(pi*x)+sin(pi*x)", -2, 1, 0.01)
test_2 = ("f(x)=-197*sin(sqrt(abs(x/2+197)))-x*sin(sqrt(abs(x-197)))", -360, 180, 0.01)

def main(test):
    func = test[0]
    x_start = test[1]
    x_end = test[2]
    eps = test[3]
    pm = PiyavskyMethod(func, x_start, x_end, eps)
    x_min, f_min = pm.solve()
    print(f"Найденный минимум: f({x_min})={f_min}")
    print(f"Число итераций: {pm.iteration_count}")
    pm.plot()


if __name__ == "__main__":
    main(test_2)
