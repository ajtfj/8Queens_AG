import random
import numpy as np
import matplotlib.pyplot as plt

class EightQueens:
    population = []

    def __init__(self) -> None:
        pass

    def fenotype_to_chromosome(self, fenotype: list[int]):
        chromosome = ''
        for f in fenotype:
            chromosome += format(f, '03b')

        return chromosome
    
    def chromosome_to_fenotype(self, chromosome: str):
        fenotype = []
        for i in range(0, len(chromosome)-1, 3):
            decimal = int(chromosome[i:i+3], 2)
            fenotype.append(decimal)

        return fenotype
    
    def cut_and_crossfill(self, parents: list[str]):
        if random.uniform(0,1) <= 0.9:
            parent1 = parents[0]
            parent2 = parents[1]
            cut_point = random.randint(0,7) * 3

            child1 = parent1[:cut_point]
            child2 = parent2[:cut_point]
            child1 = self.crossfill(child1, parent2, cut_point)
            child2 = self.crossfill(child2, parent1, cut_point)
            if random.uniform(0,1) <= 0.4:
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
        else:
            child1 = parents[0]
            child2 = parents[1]
        
        return [child1,child2]
    
    def crossfill(self, child: str, parent: str, cut_point: int):
        index = cut_point
        while len(child) < 24:
            if not self.isIn(child,parent[index:index+3]):
                child += parent[index:index+3]
            index = (index+3)%24
        return child

    def isIn(self, child: str, value: str)-> bool:
        for i in range(0, len(child)-1, 3):
            if value == child[i:i+3]:
                return True
        return False
    
    def mutate(self, child: str):
            a, b = [p*3 for p in sorted(random.sample(range(8), 2))]

            child = child[:a] + child[b:b+3] + child[a+3:b] + child[a:a+3] + child[b+3:]
            return child

    def generate_population(self, size: int):
        self.population = []
        fenotype = [0,1,2,3,4,5,6,7]

        for _ in range(size):
            random.shuffle(fenotype)
            chromosome = self.fenotype_to_chromosome(fenotype)
            self.population.append(chromosome)

    def select_parents(self, population: list[list[str|float|int]]):
        positions = random.sample(range(len(self.population)), 5)
        parents = []
        for position in positions:
            parents.append(population[position])
        parents.sort(key=lambda tup: tup[1], reverse=True)
        selected_parents = parents[:2]
        return list(map(lambda tup: tup[0], selected_parents))

    def calcule_fitness(self, chromosome: str):
        fenotype = self.chromosome_to_fenotype(chromosome)
        penalty = 0

        for x_col, x_row in enumerate(fenotype):
            for y_col in range(x_col+1, 8):
                y_row = fenotype[y_col]
                if y_col - y_row == x_col - x_row or y_col + y_row == x_col + x_row:
                    penalty += 1
        
        return 1/(1+penalty)
    
    def score(self,sample=None):
        if sample == None:
            sample = self.population[:]

        fitness = [self.calcule_fitness(chromosome) for chromosome in sample]
        score = [list(s) for s in zip(sample, fitness, range(len(sample)))]
        score.sort(key=lambda item: item[1], reverse=True)

        return score
    
    def survivors_select(self, children:list[str]):
        for child in children:
            self.population.append(child)
        score = self.score()
        worts = []
        for i in range(len(children)):
            worts.append(score[len(self.population)-1-i][2])
            worts.sort(reverse=True)
        for w in worts:
            self.population.pop(w)

    def solution(self):
        score = self.score()
        if score[0][1] == 1:
            return score[0]
        return None

def run(eightQueens: EightQueens):
    eightQueens.generate_population(100)
    population_fitness = eightQueens.score()
    solution = eightQueens.solution()
    evaluation = 0
    best_fitness_history = [population_fitness[0][1]]
    average_fitness_history = [np.mean(list(map(lambda x: x[1], population_fitness)))]

    while solution == None and evaluation < 10000:
        parents = eightQueens.select_parents(population_fitness)
        children = eightQueens.cut_and_crossfill(parents)
        eightQueens.survivors_select(children)
        population_fitness = eightQueens.score()
        solution = eightQueens.solution()
        best_fitness_history.append(population_fitness[0][1])
        average_fitness_history.append(np.mean(list(map(lambda x: x[1], population_fitness))))
        evaluation += 1
    
    totalConverged = len(list(filter(lambda x: x[1] == 1, population_fitness)))

    return totalConverged, evaluation, best_fitness_history, average_fitness_history

def find_all(eightQueens: EightQueens, evaluation=0, best_fitness_history=[], average_fitness_history=[]):
    population_fitness = eightQueens.score()

    while population_fitness[-1][1] < 1.0:
        parents = eightQueens.select_parents(population_fitness)
        children = eightQueens.cut_and_crossfill(parents)
        eightQueens.survivors_select(children)
        population_fitness = eightQueens.score()
        evaluation += 1
    
    return evaluation