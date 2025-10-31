import math
import numpy as np


class PiyavskyMethod:
    """
    Класс для нахождения минимума липшицевой функции с помощью метода Пиявского (ломанных).
    """
    def __init__(self, func: str, x_start: int, x_end: int, eps: float) -> None:
        """
        Метод для инициализации класса. Принимает условия задачи и осуществляет первичную обработку.

        Аргументы:
            func: Строка с липшицевой функцией (например, "f(x)=x+sin(3.14159*x)").
            x_start: Координата X начала рассматриваемого отрезка.
            x_end: Координата X конца рассматриваемого отрезка.
            eps: Точность вычислений (например, 0.01).

        Поддерживаемые операторы:
            sin, cos, tan, exp, log, sqrt, pi, e
            (При необходимости может быть реализована поддержка дополнительных операторов)
        """
        # Сохраняем заданные параметры на уровне класса
        self.func_str = func
        self.x_start = x_start
        self.x_end = x_end
        self.eps = eps

        # Получаем программное представление заданной функции
        # (чтобы можно было находить значения функции при различных X)
        self.func = self._create_func()

        # Рассчитываем и получаем значение константы Липшица
        self.L = self._get_L()
    
    def _create_func(self):
        """
        Метод для получения программного представления заданной функции.
        """
        # Преобразуем строку с функцией к читаемому для eval формату
        # (при необходимости можно добавить обработку дополнительных операторов)
        func_expr = self.func_str
        func_expr = func_expr.replace("f(x)=", "")
        func_expr = func_expr.replace("sin", "math.sin")
        func_expr = func_expr.replace("cos", "math.cos")
        func_expr = func_expr.replace("tan", "math.tan")
        func_expr = func_expr.replace("exp", "math.exp")
        func_expr = func_expr.replace("log", "math.log")
        func_expr = func_expr.replace("sqrt", "math.sqrt")
        func_expr = func_expr.replace("pi", "math.pi")
        func_expr = func_expr.replace("e", "math.e")

        # Возвращаем лямбда-функцию
        return lambda x: eval(func_expr, {"math": math, "x": x})

    def _get_L(self) -> float:
        """
        Метод для расчёта приблизительного значения константы Липшица L.
        """
        # Разбиваем отрезок на равномерно распределённые точки
        n_points = 1000
        x_samples = np.linspace(self.x_start, self.x_end, n_points)

        # Начальное значение максимальной производной
        max_deriv = 0

        # Рассчитываем шаг для расчёта производных
        h = x_samples[1] - x_samples[0]

        # Перебираем все точки
        for x in x_samples:
            # Проверяем, что мы не выходим за пределы заданного диапазона
            if x - h >= self.x_start and x + h <= self.x_end:
                # Рассчитываем производную для рассматриваемого X
                derivative = (self.func(x + h) - self.func(x - h)) / (2 * h)
                # При необходимости обновляем максимальное значение производной
                max_deriv = max(max_deriv, abs(derivative))
        
        # Возвращаем максимальное значение производной с небольшим запасом
        # (для большей надёжности)
        return max_deriv * 1.1
