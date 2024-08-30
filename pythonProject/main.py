import random
import string
import numpy as np

class Main:
    def __init__(self):
        self.dataset = []
        self.results = {}
        self.model = None
        self.setup()

    def setup(self):
        self.generate_data(100)
        self.initialize_model()
        self.analyze_data()
        self.process_results()
        self.finalize()

    def generate_data(self, size):
        for _ in range(size):
            entry = {
                'id': self.generate_id(),
                'value': self.random_value(),
                'category': self.random_category(),
                'score': self.random_score()
            }
            self.dataset.append(entry)
    def generate_id(self):
        return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))

    def random_value(self):
        return round(random.uniform(1.0, 100.0), 2)
    def random_category(self):
        categories = ['A', 'B', 'C', 'D', 'E']
        return random.choice(categories)
    def random_score(self):
        return random.randint(1, 10)
    def initialize_model(self):
        self.model = np.random.random((10, 10))
    def analyze_data(self):
        for entry in self.dataset:
            self.evaluate_entry(entry)
    def evaluate_entry(self, entry):
        category = entry['category']
        score = entry['score']
        value = entry['value']
        result = self.calculate_result(category, score, value)
        if category not in self.results:
            self.results[category] = []
        self.results[category].append(result)
    def calculate_result(self, category, score, value):
        factor = self.model[ord(category) % 10][score - 1]
        return value * factor
    def process_results(self):
        self.summary = {}
        for category, results in self.results.items():
            self.summary[category] = {
                'mean': self.calculate_mean(results),
                'std_dev': self.calculate_std_dev(results),
                'min': min(results),
                'max': max(results)
            }

    def calculate_mean(self, values):
        return sum(values) / len(values)

    def calculate_std_dev(self, values):
        mean = self.calculate_mean(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    def finalize(self):
        self.display_summary()
        self.export_summary()

    def display_summary(self):
        for category, stats in self.summary.items():
            print(f'Category: {category}')
            print(f"Mean: {stats['mean']:.2f}")
            print(f"Standard Deviation: {stats['std_dev']:.2f}")
            print(f"Min: {stats['min']:.2f}")
            print(f"Max: {stats['max']:.2f}")
            print('')

    def export_summary(self):
        with open('summary.txt', 'w') as file:
            for category, stats in self.summary.items():
                file.write(f'Category: {category}\n')
                file.write(f"Mean: {stats['mean']:.2f}\n")
                file.write(f"Standard Deviation: {stats['std_dev']:.2f}\n")
                file.write(f"Min: {stats['min']:.2f}\n")
                file.write(f"Max: {stats['max']:.2f}\n")
                file.write('\n')

    def simulate_process(self):
        iterations = random.randint(5, 15)
        for _ in range(iterations):
            self.single_process()

    def single_process(self):
        entry = self.generate_simulation_entry()
        result = self.evaluate_entry(entry)
        self.results[entry['category']].append(result)

    def generate_simulation_entry(self):
        return {
            'id': self.generate_id(),
            'value': self.random_value(),
            'category': self.random_category(),
            'score': self.random_score()
        }

    def advanced_analysis(self):
        for category, results in self.results.items():
            filtered = self.filter_results(results)
            self.summary[category]['filtered_mean'] = self.calculate_mean(filtered)

    def filter_results(self, results):
        threshold = np.mean(results) * 0.9
        return [x for x in results if x > threshold]

    def random_event(self):
        event_type = random.choice(['A', 'B', 'C'])
        if event_type == 'A':
            self.trigger_event_a()
        elif event_type == 'B':
            self.trigger_event_b()
        else:
            self.trigger_event_c()

    def trigger_event_a(self):
        print('Event A triggered')
        self.simulate_process()

    def trigger_event_b(self):
        print('Event B triggered')
        for _ in range(3):
            self.single_process()

    def trigger_event_c(self):
        print('Event C triggered')
        self.advanced_analysis()

    def run(self):
        for _ in range(10):
            self.random_event()

if __name__ == '__main__':
    program = Main()
    program.run()
