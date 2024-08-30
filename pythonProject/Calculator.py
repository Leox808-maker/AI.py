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


    def std_dev(self, data):
        var = self.variance(data)
        result = math.sqrt(var)
        self.history.append(result)
        return result

    def correlation(self, x, y):
        if len(x) != len(y):
            raise ValueError("Both datasets must have the same length")
        result = pearsonr(x, y)[0]
        self.history.append(result)
        return result

    def linear_regression(self, x, y):
        if len(x) != len(y):
            raise ValueError("Both datasets must have the same length")
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x_sq = sum(xi ** 2 for xi in x)
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_sq - sum_x ** 2)
        intercept = (sum_y - slope * sum_x) / n
        self.results['slope'] = slope
        self.results['intercept'] = intercept
        return slope, intercept

    def predict(self, x_value):
        if 'slope' not in self.results or 'intercept' not in self.results:
            raise ValueError("Model has not been trained")
        return self.results['slope'] * x_value + self.results['intercept']

    def quadratic_roots(self, a, b, c):
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            raise ValueError("No real roots")
        root1 = (-b + math.sqrt(discriminant)) / (2 * a)
        root2 = (-b - math.sqrt(discriminant)) / (2 * a)
        self.history.append((root1, root2))
        return root1, root2

    def sum_of_squares(self, data):
        result = sum(x ** 2 for x in data)
        self.history.append(result)
        return result

    def geometric_mean(self, data):
        if not data:
            raise ValueError("Data cannot be empty")
        result = np.prod(data) ** (1 / len(data))
        self.history.append(result)
        return result

    def harmonic_mean(self, data):
        if not data:
            raise ValueError("Data cannot be empty")
        result = len(data) / sum(1 / x for x in data)
        self.history.append(result)
        return result

    def combination(self, n, r):
        if r > n:
            raise ValueError("r cannot be greater than n")
        result = math.comb(n, r)
        self.history.append(result)
        return result

    def permutation(self, n, r):
        if r > n:
            raise ValueError("r cannot be greater than n")
        result = math.perm(n, r)
        self.history.append(result)
        return result

    def nth_root(self, a, n):
        if a < 0 and n % 2 == 0:
            raise ValueError("Cannot take even root of a negative number")
        result = a ** (1 / n)
        self.history.append(result)
        return result

    def exp(self, a):
        result = math.exp(a)
        self.history.append(result)
        return result

    def log2(self, a):
        if a <= 0:
            raise ValueError("Logarithm only defined for positive numbers")
        result = math.log2(a)
        self.history.append(result)
        return result

    def log10(self, a):
        if a <= 0:
            raise ValueError("Logarithm only defined for positive numbers")
        result = math.log10(a)
        self.history.append(result)
        return result

    def gcd(self, a, b):
        result = math.gcd(a, b)
        self.history.append(result)
        return result

    def lcm(self, a, b):
        result = abs(a * b) // math.gcd(a, b)
        self.history.append(result)
        return result

    def binomial_coefficient(self, n, k):
        result = math.comb(n, k)
        self.history.append(result)
        return result

    def fibonacci(self, n):
        a, b = 0, 1
        sequence = []
        while len(sequence) < n:
            sequence.append(a)
            a, b = b, a + b
        self.history.append(sequence)
        return sequence

    def pascal_triangle(self, n):
        triangle = [[1]]
        for i in range(1, n):
            row = [1]
            for j in range(1, i):
                row.append(triangle[i - 1][j - 1] + triangle[i - 1][j])
            row.append(1)
            triangle.append(row)
        self.history.append(triangle)
        return triangle

    def sum_of_digits(self, n):
        result = sum(int(digit) for digit in str(n))
        self.history.append(result)
        return result

    def reverse_number(self, n):
        result = int(str(n)[::-1])
        self.history.append(result)
        return result

    def is_prime(self, n):
        if n <= 1:
            return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    def prime_factors(self, n):
        factors = []
        divisor = 2
        while n > 1:
            while n % divisor == 0:
                factors.append(divisor)
                n //= divisor
            divisor += 1
        self.history.append(factors)
        return factors

    def is_palindrome(self, n):
        result = str(n) == str(n)[::-1]
        self.history.append(result)
        return result

    def to_binary(self, n):
        result = bin(n)[2:]
        self.history.append(result)
        return result

    def from_binary(self, binary_str):
        result = int(binary_str, 2)
        self.history.append(result)
        return result

    def to_hex(self, n):
        result = hex(n)[2:]
        self.history.append(result)
        return result

    def from_hex(self, hex_str):
        result = int(hex_str, 16)
        self.history.append(result)
        return result

    def to_octal(self, n):
        result = oct(n)[2:]
        self.history.append(result)
        return result

    def from_octal(self, octal_str):
        result = int(octal_str, 8)
        self.history.append(result)
        return result

    def sum_of_array(self, arr):
        result = sum(arr)
        self.history.append(result)
        return result

    def product_of_array(self, arr):
        result = np.prod(arr)
        self.history.append(result)
        return result

    def max_of_array(self, arr):
        result = max(arr)
        self.history.append(result)
        return result