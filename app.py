from fetal_health_KNN import FetalHealthKNN
from genetic_algorithm_knn_optmizer import run_genetic_experiment

def main():
    print(f"\n =============RESULTADO DO KNN ORIGINAL============:")
    X_train, y_train, X_test, y_test, preprocessor, best_k = FetalHealthKNN.run_fetal_health_knn()

    print(f"\n =============RESULTADO DO KNN OTIMIZADO============:")
    experiment1 = run_genetic_experiment(
        X_train, y_train, X_test, y_test, preprocessor,
        "01",
        population_size=20,
        generations=15,
        mutation_rate=0.05,
        crossover_rate=0.7
    )

if __name__ == "__main__":
    main()
