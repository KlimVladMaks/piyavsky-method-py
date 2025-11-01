from piyavsky_method import PiyavskyMethod


test_1 = ("f(x)=2*x+cos(pi*x)+sin(pi*x)", -2, 1, 0.01)
test_2 = ("f(x)=-197*sin(sqrt(abs(x/2+197)))-x*sin(sqrt(abs(x-197)))", -360, 180, 0.01)
test_3 = ("f(x)=sqrt(x)*sin(x)", 1, 8, 0.01)
test_4 = ("f(x)=sqrt(1+3*cos(x)**2)+cos(10*x)", 1, 5, 0.01)
test_5 = ("f(x)=100+cos(10*x)", 1, 3, 0.01)
# Функция Растригина
test_6 = ("f(x)=10+x**2-10*cos(2*pi*x)", -3, 3, 0.01)
# Функция Экли
test_7 = ("f(x)=-20*exp(-0.2*sqrt(x**2))-exp(cos(2*pi*x))+20+exp(1)", -3, 3, 0.01)


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
    main(test_7)
