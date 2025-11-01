from piyavsky_method import PiyavskyMethod


test_1 = ("f(x)=2*x+math.cos(math.pi*x)+math.sin(math.pi*x)", -2, 1, 0.01)
test_2 = ("f(x)=-197*math.sin(math.sqrt(abs(x/2+197)))-x*math.sin(math.sqrt(abs(x-197)))", -360, 180, 0.01)
test_3 = ("f(x)=math.sqrt(x)*math.sin(x)", 1, 8, 0.01)
test_4 = ("f(x)=math.sqrt(1+3*math.cos(x)**2)+math.cos(10*x)", 1, 5, 0.01)
test_5 = ("f(x)=100+math.cos(10*x)", 1, 3, 0.01)
# Функция Растригина
test_6 = ("f(x)=10+x**2-10*math.cos(2*math.pi*x)", -3, 3, 0.01)
# Функция Экли
test_7 = ("f(x)=-20*math.exp(-0.2*math.sqrt(x**2))-math.exp(math.cos(2*math.pi*x))+20+math.exp(1)", -3, 3, 0.01)
test_8 = ("f(x)=abs(math.sin(x))+0.5*abs(math.cos(5*x))", -2.5, 2.5, 0.01)
test_9 = ("f(x)=math.sin(x) + 0.3*math.sin(10*x) + 0.1*math.sin(50*x) + 0.01*x**2", -5, 5, 0.01)
test_10 = ("f(x)=math.tanh(10*(x-1))-math.tanh(10*(x+1))+0.5*math.sin(5*x)", -3, 3, 0.01)
test_11 = ("f(x)=0.5*(math.sin(20*x)/(20*x)+1)+0.2*math.sin(x)+0.1*x", 0.1, 4, 0.01)
test_12 = ("f(x)=math.sin(5*x)*math.exp(-abs(x))+0.3*math.cos(15*x)*math.exp(-0.5*abs(x-2))+0.05*x", -3, 5, 0.01)


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
    main(test_12)
