from fetal_health_KNN import FetalHealthKNN
from genetic_algorithm_knn_optmizer import GeneticAlgorithmKnnOptimizer
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from joblib import dump

def run_knn_original():
    df = FetalHealthKNN.data_treatment() 
    X, y, X_train, X_test, y_train, y_test, X_resampled, y_resampled, preprocessor = FetalHealthKNN.data_preprocessing(df)
    accuracy_knn, recall_knn, f1_knn, knn_report, y_pred_knn, y_test, knn_pipeline = FetalHealthKNN.run_fetal_health_knn(X_resampled, y_resampled, X_test, y_test, preprocessor)
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
    return best_experiment, experiments

def compare_original_vs_optimized(original_results, best_experiment):
    accuracy_orig, recall_orig, f1_orig = original_results[0], original_results[1], original_results[2]
    
    categories = ['Accuracy', 'Recall', 'F1-Score']
    original_values = [accuracy_orig, recall_orig, f1_orig]
    optimized_values = [
        best_experiment['best_metrics']['accuracy'],
        best_experiment['best_metrics']['recall'],
        best_experiment['best_metrics']['f1_score']
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, original_values, width, label='KNN Original', color='lightblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, optimized_values, width, label='KNN Otimizado', color='lightgreen', alpha=0.8)
    
    ax1.set_xlabel('Métricas')
    ax1.set_ylabel('Valores')
    ax1.set_title('Comparação: KNN Original vs Otimizado', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    improvements = []
    for orig, opt in zip(original_values, optimized_values):
        improvement = ((opt - orig) / orig) * 100
        improvements.append(improvement)
    
    colors = ['green' if imp >= 0 else 'red' for imp in improvements]
    bars3 = ax2.bar(categories, improvements, color=colors, alpha=0.7)
    
    ax2.set_xlabel('Métricas')
    ax2.set_ylabel('Melhoria (%)')
    ax2.set_title('Melhoria Percentual: Otimizado vs Original', fontsize=14, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, imp in zip(bars3, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.3),
                f'{imp:+.2f}%', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def main():
    # Executa o KNN original
    accuracy_knn, recall_knn, f1_knn, knn_report, y_pred_knn, y_test = run_knn_original()
    print(f"\n =============RESULTADO DO KNN ORIGINAL============:")
    FetalHealthKNN.show_results(accuracy_knn, recall_knn, f1_knn, knn_report, y_pred_knn, y_test)

    # Executa os experimentos de otimização
    print(f"\n")
    best_experiment, all_experiments = run_knn_optimized()
    print(f"\n =============RESULTADO KNN OTIMIZADO COM MELHOR EXPERIMENTO============:")
    print(f"\n")
    print(f"Melhor Experimento: {best_experiment['experiment_number']}")
    print(f"Melhor Fitness: {best_experiment['best_fitness']:.4f}")
    print(f"Melhor Accuracy: {best_experiment['best_metrics']['accuracy']:.4f}")
    print(f"Melhor Recall: {best_experiment['best_metrics']['recall']:.4f}")
    print(f"Melhor F1-Score: {best_experiment['best_metrics']['f1_score']:.4f}")
    print(f"Melhores Parâmetros: {best_experiment['best_params']}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dump(best_experiment['knn_pipeline'], f"data/modelo_classificacaofetal{best_experiment['experiment_number']}_{timestamp}.joblib")
    print("Modelo salvo com sucesso!")
    
    # Gráfico de comparação original vs otimizado
    original_results = (accuracy_knn, recall_knn, f1_knn)
    compare_original_vs_optimized(original_results, best_experiment)

if __name__ == "__main__":
    main()
