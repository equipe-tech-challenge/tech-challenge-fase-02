from fetal_health_KNN import FetalHealthKNN
from genetic_algorithm_knn_optmizer import GeneticAlgorithmKnnOptimizer

def run_knn_original():
    df = FetalHealthKNN.data_treatment() 
    X, y, X_train, X_test, y_train, y_test, X_resampled, y_resampled, preprocessor = FetalHealthKNN.data_preprocessing(df)
    accuracy_knn, recall_knn, f1_knn, knn_report, y_pred_knn, y_test = FetalHealthKNN.run_fetal_health_knn(X_resampled, y_resampled, X_test, y_test, preprocessor)
    return accuracy_knn, recall_knn, f1_knn, knn_report, y_pred_knn, y_test

def run_knn_optimized():
    df = FetalHealthKNN.data_treatment() 
    X, y, X_train, X_test, y_train, y_test, X_resampled, y_resampled, preprocessor = FetalHealthKNN.data_preprocessing(df)

    print(f"\n =============RESULTADO DO KNN OTIMIZADO============:")
    experiments = []
    experiment1 = GeneticAlgorithmKnnOptimizer.run_genetic_experiment(
        X_train, y_train, X_test, y_test, preprocessor, X_resampled, y_resampled,
        "01",
        population_size=20,
        generations=15,
        mutation_rate=0.05,
        crossover_rate=0.7
    )
    experiments.append(experiment1)

    experiment2 = GeneticAlgorithmKnnOptimizer.run_genetic_experiment(
        X_train, y_train, X_test, y_test, preprocessor, X_resampled, y_resampled,
        "02",
        population_size=30,
        generations=20,
        mutation_rate=0.1,
        crossover_rate=0.8
    )
    experiments.append(experiment2)

    experiment3 = GeneticAlgorithmKnnOptimizer.run_genetic_experiment(
        X_train, y_train, X_test, y_test, preprocessor, X_resampled, y_resampled,
        "03",
        population_size=50,
        generations=25,
        mutation_rate=0.15,
        crossover_rate=0.9
    )
    experiments.append(experiment3)

    best_experiment = max(experiments, key=lambda x: x['best_fitness'])
    return best_experiment

def main():
    accuracy_knn, recall_knn, f1_knn, knn_report, y_pred_knn, y_test = run_knn_original()
    print(f"\n =============RESULTADO DO KNN ORIGINAL============:")
    FetalHealthKNN.show_results(accuracy_knn, recall_knn, f1_knn, knn_report, y_pred_knn, y_test)

    print(f"\n")
    best_experiment = run_knn_optimized()
    print(f"\n =============RESULTADO KNN OTIMIZADO COM MELHOR EXPERIMENTO============:")
    print(f"\n")
    print(f"Melhor Experimento: {best_experiment['experiment_number']}")
    print(f"Melhor Fitness: {best_experiment['best_fitness']:.4f}")
    print(f"Melhor Accuracy: {best_experiment['best_metrics']['accuracy']:.4f}")
    print(f"Melhor Recall: {best_experiment['best_metrics']['recall']:.4f}")
    print(f"Melhor F1-Score: {best_experiment['best_metrics']['f1_score']:.4f}")
    print(f"Melhores Parâmetros: {best_experiment['best_params']}")

if __name__ == "__main__":
    main()
