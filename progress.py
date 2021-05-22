import math


class Progress:

    def __init__(self, total_iterationss):
        self.total_iterations = total_iterationss
        self.current_iteration = 0
        self.progress = 0

    def log(self):
        self.current_iteration += 1
        current_progress = math.floor(self.current_iteration / self.total_iterations * 100)
        if self.progress < current_progress:
            self.progress = current_progress
            print(".", end="")

        if self.current_iteration >= self.total_iterations:
            print()
