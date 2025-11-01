import math
import numpy as np


class PiyavskyMethod:
    """
    Класс для нахождения минимума липшицевой функции с помощью метода Пиявского (ломанных).
    """
    def __init__(self, func: str, x_start: float, x_end: float, eps: float) -> None:
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

        # В качестве двух начальных точек берём точки границ отрезка 
        self.points = [
            (self.x_start, self.func(self.x_start)),
            (self.x_end, self.func(self.x_end))
        ]

    def piyavsky_step(self) -> bool:
        """
        Делает один шаг по методу Пиявского.

        Returns:
            True если достигнута заданная точность, False если ещё нет.
        """
        # Находим следующую точку для исследования
        x_new, lower_bound = self._find_next_point()

        # Вычисляем реальное значение функции
        f_new = self.func(x_new)

        # Добавляем новую точку
        self.points.append((x_new, f_new))

        # Сортируем список точек по X
        self.points.sort(key=lambda p: p[0])

        # Находим текущий зазор
        current_best_f = min(p[1] for p in self.points)
        gap = current_best_f - lower_bound

        # Если зазор меньше заданной точности, то возвращаем True, если больше - False
        return gap < self.eps

    def _find_next_point(self) -> tuple[float, float]:
        """
        Находит следующую точку для исследования и значение нижней оценки.
        
        Returns:
            (x_new, lower_bound) - координата X новой точки и значение ломаной в ней
        """
        best_x = None
        best_lower_bound = float('inf')

        # Проверяем все интервалы между соседними точками
        for i in range(len(self.points) - 1):
            x1, f1 = self.points[i]
            x2, f2 = self.points[i + 1]

            # Вычисляем координаты точки пересечения
            x_intersect = (f1 - f2 + self.L * (x1 + x2)) / (2 * self.L)
            y_intersect = f1 - self.L * (x_intersect - x1)

            # При необходимости обновляем значение нижней границы
            if y_intersect < best_lower_bound:
                best_lower_bound = y_intersect
                best_x = x_intersect
        
        # Возвращаем полученные значения
        return best_x, best_lower_bound

    def _get_intersect_coord(self, x_1: int, x_2: int) -> tuple[int]:
        """
        Возвращает координаты X и Y точки пересечения двух прямых с наклонами -L и L,
        проходящих через две заданные точки на графике функции с координатами x_1 и x_2.
        Для корректной работы обязательно, чтобы x_1 была меньше x_2.
        """
        f_1 = self.func(x_1)
        f_2 = self.func(x_2)
        x_intersect = ((f_1 - f_2) / (2 * self.L)) + ((x_1 + x_2) / 2)
        y_intersect = f_1 - self.L * (x_intersect - x_1)
        return x_intersect, y_intersect
    
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
