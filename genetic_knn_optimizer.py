import random
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

class GeneticKnnOptimizer:
    def __init__(self, X_train, y_train, X_test, y_test, preprocessor, 
                 population_size=30, generations=20, mutation_rate=0.1, 
                 crossover_rate=0.8, elite_size=2):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.preprocessor = preprocessor
        
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        
        self.fitness_history = []
        self.best_individual_history = []
        
    def create_individual(self):
        n_neighbors = random.randint(1, 20)
        weights = random.choice(['uniform', 'distance'])
        metric = random.choice(['euclidean', 'manhattan', 'minkowski'])
        algorithm = random.choice(['auto', 'ball_tree', 'kd_tree', 'brute'])
        leaf_size = random.randint(10, 50)
        
        return {
            'n_neighbors': n_neighbors,
            'weights': weights,
            'metric': metric,
            'algorithm': algorithm,
            'leaf_size': leaf_size
        }
    
    def create_population(self):
        return [self.create_individual() for _ in range(self.population_size)]
    
    def fitness_function(self, individual):
        try:
            knn = KNeighborsClassifier(
                n_neighbors=individual['n_neighbors'],
                weights=individual['weights'],
                metric=individual['metric'],
                algorithm=individual['algorithm'],
                leaf_size=individual['leaf_size']
            )
            
            pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('classifier', knn)
            ])
            
            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)
            
            accuracy = accuracy_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred, average='macro')
            f1 = f1_score(self.y_test, y_pred, average='macro')
            
            fitness = (accuracy * 0.4 + recall * 0.3 + f1 * 0.3)
            
            return fitness, {
                'accuracy': accuracy,
                'recall': recall,
                'f1_score': f1,
                'individual': individual
            }
        except Exception as e:
            return 0.0, {
                'accuracy': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'individual': individual,
                'error': str(e)
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
            
            if random.random() < 0.5:
                child1['algorithm'] = parent2['algorithm']
                child2['algorithm'] = parent1['algorithm']
            
            if random.random() < 0.5:
                child1['leaf_size'] = parent2['leaf_size']
                child2['leaf_size'] = parent1['leaf_size']
            
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
            
            if random.random() < 0.2:
                mutated['algorithm'] = random.choice(['auto', 'ball_tree', 'kd_tree', 'brute'])
            
            if random.random() < 0.2:
                mutated['leaf_size'] = random.randint(10, 50)
        
        return mutated
    
    def get_elite(self, population_with_fitness, n_elites):
        sorted_pop = sorted(population_with_fitness, key=lambda x: x['fitness'], reverse=True)
        return sorted_pop[:n_elites]
    
    def optimize(self):
        print(f"=== INICIANDO OTIMIZAÇÃO GENÉTICA ===")
        print(f"População: {self.population_size}, Gerações: {self.generations}")
        print(f"Taxa de mutação: {self.mutation_rate}, Taxa de crossover: {self.crossover_rate}")
        
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
    
    def plot_evolution(self):
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(self.fitness_history) + 1), self.fitness_history, 'b-', linewidth=2)
        plt.title('Evolução do Fitness - Algoritmo Genético')
        plt.xlabel('Geração')
        plt.ylabel('Fitness')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def get_best_parameters_history(self):
        return self.best_individual_history

def run_genetic_experiment(X_train, y_train, X_test, y_test, preprocessor, 
                          experiment_name, population_size, generations, 
                          mutation_rate, crossover_rate):
    print(f"\n{'='*60}")
    print(f"EXPERIMENTO: {experiment_name}")
    print(f"{'='*60}")
    
    optimizer = GeneticKnnOptimizer(
        X_train, y_train, X_test, y_test, preprocessor,
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate
    )
    
    best_params, best_fitness, best_metrics = optimizer.optimize()
    
    print(f"\n=== RESULTADOS DO EXPERIMENTO {experiment_name} ===")
    print(f"Melhor Fitness: {best_fitness:.4f}")
    print(f"Melhor Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Melhor Recall: {best_metrics['recall']:.4f}")
    print(f"Melhor F1-Score: {best_metrics['f1_score']:.4f}")
    print(f"Melhores Parâmetros: {best_params}")
    
    optimizer.plot_evolution()
    
    return {
        'experiment_name': experiment_name,
        'best_params': best_params,
        'best_fitness': best_fitness,
        'best_metrics': best_metrics,
        'fitness_history': optimizer.fitness_history,
        'optimizer': optimizer
    }

def compare_with_original_knn(X_train, y_train, X_test, y_test, preprocessor, 
                             best_params, original_k=3):
    print(f"\n{'='*60}")
    print("COMPARAÇÃO: MODELO ORIGINAL vs OTIMIZADO")
    print(f"{'='*60}")
    
    original_knn = KNeighborsClassifier(n_neighbors=original_k)
    original_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', original_knn)
    ])
    
    optimized_knn = KNeighborsClassifier(
        n_neighbors=best_params['n_neighbors'],
        weights=best_params['weights'],
        metric=best_params['metric'],
        algorithm=best_params['algorithm'],
        leaf_size=best_params['leaf_size']
    )
    optimized_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', optimized_knn)
    ])
    
    original_pipeline.fit(X_train, y_train)
    optimized_pipeline.fit(X_train, y_train)
    
    original_pred = original_pipeline.predict(X_test)
    optimized_pred = optimized_pipeline.predict(X_test)
    
    original_accuracy = accuracy_score(y_test, original_pred)
    original_recall = recall_score(y_test, original_pred, average='macro')
    original_f1 = f1_score(y_test, original_pred, average='macro')
    
    optimized_accuracy = accuracy_score(y_test, optimized_pred)
    optimized_recall = recall_score(y_test, optimized_pred, average='macro')
    optimized_f1 = f1_score(y_test, optimized_pred, average='macro')
    
    print(f"MODELO ORIGINAL (K={original_k}):")
    print(f"  Accuracy: {original_accuracy:.4f}")
    print(f"  Recall: {original_recall:.4f}")
    print(f"  F1-Score: {original_f1:.4f}")
    
    print(f"\nMODELO OTIMIZADO:")
    print(f"  Accuracy: {optimized_accuracy:.4f}")
    print(f"  Recall: {optimized_recall:.4f}")
    print(f"  F1-Score: {optimized_f1:.4f}")
    
    print(f"\nMELHORIAS:")
    print(f"  Accuracy: {((optimized_accuracy - original_accuracy) / original_accuracy * 100):+.2f}%")
    print(f"  Recall: {((optimized_recall - original_recall) / original_recall * 100):+.2f}%")
    print(f"  F1-Score: {((optimized_f1 - original_f1) / original_f1 * 100):+.2f}%")
    
    return {
        'original': {
            'accuracy': original_accuracy,
            'recall': original_recall,
            'f1_score': original_f1,
            'params': {'n_neighbors': original_k}
        },
        'optimized': {
            'accuracy': optimized_accuracy,
            'recall': optimized_recall,
            'f1_score': optimized_f1,
            'params': best_params
        }
    }
