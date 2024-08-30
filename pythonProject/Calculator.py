import math
import numpy as np
from collections import Counter
from scipy.stats import pearsonr

class Calculator:
    def __init__(self):
        self.history = []
        self.results = {}

    def add(self, a, b):
        result = a + b
        self.history.append(result)
        return result

    def subtract(self, a, b):
        result = a - b
        self.history.append(result)
        return result

    def multiply(self, a, b):
        result = a * b
        self.history.append(result)
        return result

    def divide(self, a, b):
        if b == 0:
            raise ValueError("Division by zero")
        result = a / b
        self.history.append(result)
        return result

    def power(self, a, b):
        result = a ** b
        self.history.append(result)
        return result

    def factorial(self, n):
        if n < 0:
            raise ValueError("Negative values not allowed")
        result = math.factorial(n)
        self.history.append(result)
        return result

    def sqrt(self, a):
        if a < 0:
            raise ValueError("Cannot take square root of a negative number")
        result = math.sqrt(a)
        self.history.append(result)
        return result

    def log(self, a, base=10):
        if a <= 0:
            raise ValueError("Logarithm only defined for positive numbers")
        result = math.log(a, base)
        self.history.append(result)
        return result

    def sin(self, a):
        result = math.sin(a)
        self.history.append(result)
        return result

    def cos(self, a):
        result = math.cos(a)
        self.history.append(result)
        return result

    def tan(self, a):
        result = math.tan(a)
        self.history.append(result)
        return result

    def mean(self, data):
        if not data:
            raise ValueError("Data cannot be empty")
        result = sum(data) / len(data)
        self.history.append(result)
        return result

    def median(self, data):
        if not data:
            raise ValueError("Data cannot be empty")
        sorted_data = sorted(data)
        n = len(sorted_data)
        mid = n // 2
        if n % 2 == 0:
            result = (sorted_data[mid - 1] + sorted_data[mid]) / 2
        else:
            result = sorted_data[mid]
        self.history.append(result)
        return result

    def mode(self, data):
        if not data:
            raise ValueError("Data cannot be empty")
        counter = Counter(data)
        mode_data = counter.most_common()
        max_count = mode_data[0][1]
        modes = [val for val, count in mode_data if count == max_count]
        self.history.append(modes)
        return modes

    def variance(self, data):
        if len(data) < 2:
            raise ValueError("Variance requires at least two data points")
        mean_value = self.mean(data)
        result = sum((x - mean_value) ** 2 for x in data) / len(data)
        self.history.append(result)
        return result