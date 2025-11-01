import math
import numpy as np
import matplotlib.pyplot as plt


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

        # Словарь для хранения кэша с точками пересечения
        # (чтобы не рассчитывать их повторно)
        self.intersect_cache = {}

        # Число итераций для достижения решения с заданной точностью.
        # Данное значение доступно лишь после полного решения задачи (до этого равно None)
        self.iteration_count = None

    def step(self) -> bool:
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

    def plot(self) -> None:
        """
        Визуализирует текущее состояние решения: функцию, вычисленные точки и ломаную.
        """
        # Создаём точки для гладкого отображения функции
        x_smooth = np.linspace(self.x_start, self.x_end, 1000)
        y_smooth = [self.func(x) for x in x_smooth]

        # Разделяем сохранённые точки на X и Y
        x_points = [p[0] for p in self.points]
        y_points = [p[1] for p in self.points]

        # Списки для координат, задающих ломаную (нижнюю огибающую)
        x_polyline = []
        y_polyline = []

        # Перебираем все интервалы
        for i in range(len(self.points) - 1):
            x1, f1 = self.points[i]
            x2, f2 = self.points[i + 1]

            # Добавляем левую точку "зубца"
            x_polyline.append(x1)
            y_polyline.append(f1)

            # Добавляем точку пересечения (вершину "зубца")
            x_intersect, y_intersect = self._get_intersect_coord(x1, f1, x2, f2)
            x_polyline.append(x_intersect)
            y_polyline.append(y_intersect)
        
        # Добавляем последнюю точку
        x_polyline.append(self.points[-1][0])
        y_polyline.append(self.points[-1][1])

        # Создаём график
        plt.figure(figsize=(12, 6))

        # График исходной функции
        plt.plot(x_smooth, y_smooth, 'b-', linewidth=2, label='Функция f(x)')
        
        # Ломаная Пиявского
        plt.plot(x_polyline, y_polyline, 'r-', linewidth=1, label='Ломаная (нижняя оценка)')
        
        # Вычисленные точки
        plt.scatter(x_points, y_points, color='red', s=50, zorder=5, label='Вычисленные точки')
        
        # Лучшая точка (минимальная)
        best_point = min(self.points, key=lambda p: p[1])
        plt.scatter([best_point[0]], [best_point[1]], color='green', s=100, zorder=10,
                    label=f'Минимальная точка: f({best_point[0]:.3f}) = {best_point[1]:.3f}')
        
        # Настройки графика
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Метод Пиявского - График текущего состояния')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Отображаем график
        plt.tight_layout()
        plt.show()

    def solve(self) -> tuple[float, float]:
        """
        Метод для решения задачи нахождения минимума функции с помощью метода Пиявского.
        Возвращает координаты найденной минимальной точки: (x_min, f(x_min))
        """
        # Число итераций
        iteration = 0

        # Максимальное число итераций
        # (защита от бесконечного цикла)
        max_iterations = 1000

        # Делаем шаги по методу Пиявского, пока не будет достигнута заданная точность
        while iteration < max_iterations:
            iteration += 1
            is_sufficient_accuracy = self.step()
            if is_sufficient_accuracy:
                break
        
        # Сохраняем число итераций
        self.iteration_count = iteration
        
        # Находим и возвращаем минимальную точку
        min_point = min(self.points, key=lambda p: p[1])
        return min_point

    def _find_next_point(self) -> tuple[float, float]:
        """
        Находит следующую точку для исследования и значение нижней оценки.
        
        Returns:
            (x_new, lower_bound) - координата X новой точки и значение ломаной в ней
        """
        best_x = None
        best_lower_bound = float('inf')

        # Начало и конец интервала, содержащего лучшую точку
        # (используется для удаления лишнего кэша)
        best_interval_x1 = None
        best_interval_x2 = None

        # Проверяем все интервалы между соседними точками
        for i in range(len(self.points) - 1):
            x1, f1 = self.points[i]
            x2, f2 = self.points[i + 1]

            # Вычисляем координаты точки пересечения
            x_intersect, y_intersect = self._get_intersect_coord(x1, f1, x2, f2)

            # При необходимости кэшируем координаты точки пересечения
            if (x1, x2) not in self.intersect_cache:
                self.intersect_cache[(x1, x2)] = (x_intersect, y_intersect)

            # При необходимости обновляем лучшие значения
            if y_intersect < best_lower_bound:
                best_lower_bound = y_intersect
                best_x = x_intersect
                best_interval_x1 = x1
                best_interval_x2 = x2
        
        # Так как этот интервал будет разделён, то удаляем его из кэша
        del self.intersect_cache[(best_interval_x1, best_interval_x2)]
        
        # Возвращаем полученные значения
        return best_x, best_lower_bound

    def _get_intersect_coord(self, x1: float, f1: float, x2: float, f2: float) -> tuple[float, float]:
        """
        Возвращает точку пересечения двух прямых с наклоном -L и L для двух заданных точек.
        При возможности использует кэш для избежания лишних вычислений, но сам значения не кэширует.
        Для корректной работы требуется, чтобы первая точка была правее второй.
        """
        if (x1, x2) in self.intersect_cache:
            return self.intersect_cache[(x1, x2)]
        x_intersect = ((f1 - f2) / (2 * self.L)) + ((x1 + x2) / 2)
        y_intersect = f1 - self.L * (x_intersect - x1)
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
        return max_deriv * 1.2
