import random
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from fetal_health_KNN import FetalHealthKNN

class GeneticAlgorithmKnnOptimizer:
    def __init__(
        self, 
        X_train, 
        y_train, 
        X_test,
        y_test, 
        preprocessor, 
        X_resampled, 
        y_resampled, 
        population_size=30, 
        generations=20, 
        mutation_rate=0.1, 
        crossover_rate=0.8, 
        elite_size=2):

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.preprocessor = preprocessor
        self.X_resampled = X_resampled
        self.y_resampled = y_resampled
        
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        
        self.fitness_history = [] #lista para armazenar o histórico da melhor fitness
        self.best_individual_history = [] #lista para armazenar o histórico da melhor indivíduo
        
    def create_individual(self):
        n_neighbors = random.randint(1, 20)
        weights = random.choice(['uniform', 'distance'])
        metric = random.choice(['euclidean', 'manhattan', 'minkowski'])
       
        return {
            'n_neighbors': n_neighbors,
            'weights': weights,
            'metric': metric,
        }
    
    def create_population(self):
        return [self.create_individual() for _ in range(self.population_size)]
    
    def fitness_function(self, individual):
        # Usa a função do fetal_health_KNN.py para avaliar os parâmetros
        accuracy, recall, f1, knn_report, y_pred, y_test, knn_pipeline = FetalHealthKNN.run_fetal_health_knn(
            self.X_resampled, 
            self.y_resampled, 
            self.X_test, 
            self.y_test, 
            self.preprocessor,
            n_neighbors=individual['n_neighbors'],
            weights=individual['weights'],
            metric=individual['metric']
        )
        
        fitness = (accuracy * 0.4 + recall * 0.3 + f1 * 0.3)
        
        return fitness, {
            'accuracy': accuracy,
            'recall': recall,
            'f1_score': f1,
            'individual': individual
        }
    
    def tournament_selection(self, population, k=3):
        tournament = random.sample(population, k)
        best = max(tournament, key=lambda x: x['fitness'])
        return best['individual']
    
    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            child1 = parent1.copy()
            child2 = parent2.copy()
            
            if random.random() < 0.5:
                child1['n_neighbors'] = parent2['n_neighbors']
                child2['n_neighbors'] = parent1['n_neighbors']
            
            if random.random() < 0.5:
                child1['weights'] = parent2['weights']
                child2['weights'] = parent1['weights']
            
            if random.random() < 0.5:
                child1['metric'] = parent2['metric']
                child2['metric'] = parent1['metric']
            
            return child1, child2
        
        return parent1.copy(), parent2.copy()
    
    def mutation(self, individual):
        mutated = individual.copy()
        
        if random.random() < self.mutation_rate:
            if random.random() < 0.2:
                mutated['n_neighbors'] = random.randint(1, 20)
            
            if random.random() < 0.2:
                mutated['weights'] = random.choice(['uniform', 'distance'])
            
            if random.random() < 0.2:
                mutated['metric'] = random.choice(['euclidean', 'manhattan', 'minkowski'])
        
        return mutated
    
    def get_elite(self, population_with_fitness, n_elites):
        sorted_pop = sorted(population_with_fitness, key=lambda x: x['fitness'], reverse=True)
        return sorted_pop[:n_elites]
    
    def optimize(self):
        population = self.create_population()
        
        for generation in range(self.generations):
            population_with_fitness = []
            
            for individual in population:
                fitness, metrics = self.fitness_function(individual)
                population_with_fitness.append({
                    'individual': individual,
                    'fitness': fitness,
                    'metrics': metrics
                })
            
            best_individual = max(population_with_fitness, key=lambda x: x['fitness'])
            self.fitness_history.append(best_individual['fitness'])
            self.best_individual_history.append(best_individual['individual'])
            
            print(f"Geração {generation + 1}: Melhor Fitness = {best_individual['fitness']:.4f}")
            print(f"  Accuracy: {best_individual['metrics']['accuracy']:.4f}")
            print(f"  Recall: {best_individual['metrics']['recall']:.4f}")
            print(f"  F1-Score: {best_individual['metrics']['f1_score']:.4f}")
            print(f"  Parâmetros: {best_individual['individual']}")
            
            new_population = []
            
            elites = self.get_elite(population_with_fitness, self.elite_size)
            new_population.extend([elite['individual'] for elite in elites])
            
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population_with_fitness)
                parent2 = self.tournament_selection(population_with_fitness)
                
                child1, child2 = self.crossover(parent1, parent2)
                
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        final_fitness = []
        for individual in population:
            fitness, metrics = self.fitness_function(individual)
            final_fitness.append({
                'individual': individual,
                'fitness': fitness,
                'metrics': metrics
            })
        
        best_final = max(final_fitness, key=lambda x: x['fitness'])
        
        return best_final['individual'], best_final['fitness'], best_final['metrics']
    
    def run_genetic_experiment(
        X_train, 
        y_train, 
        X_test, 
        y_test, 
        preprocessor, 
        X_resampled, 
        y_resampled, 
        experiment_number, 
        population_size, 
        generations, 
        mutation_rate, 
        crossover_rate):
        print(f"EXPERIMENTO: {experiment_number}")
        
        optimizer = GeneticAlgorithmKnnOptimizer(
            X_train, y_train, X_test, y_test, preprocessor, X_resampled, y_resampled,
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate
        )
        
        best_params, best_fitness, best_metrics = optimizer.optimize()
        
        # Treina o modelo final com os melhores parâmetros
        knn_pipeline, y_pred_knn = FetalHealthKNN.train_test_knn(
            X_resampled, y_resampled, X_test, y_test, preprocessor,
            n_neighbors=best_params['n_neighbors'],
            weights=best_params['weights'],
            metric=best_params['metric']
        )
        
        print(f"\n=== RESULTADO DO KNN OTIMIZADO EXPERIMENTO {experiment_number} ===")
        print(f"Melhor Fitness: {best_fitness:.4f}")
        print(f"Melhor Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"Melhor Recall: {best_metrics['recall']:.4f}")
        print(f"Melhor F1-Score: {best_metrics['f1_score']:.4f}")
        print(f"Melhores Parâmetros: {best_params}")
        print(f"\n")

        return {
            'experiment_number': experiment_number,
            'best_params': best_params,
            'best_fitness': best_fitness,
            'best_metrics': best_metrics,
            'fitness_history': optimizer.fitness_history,
            'optimizer': optimizer,
            'knn_pipeline': knn_pipeline
        }
