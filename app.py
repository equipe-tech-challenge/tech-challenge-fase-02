import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, recall_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from imblearn.over_sampling import SMOTE
import shap
from genetic_knn_optimizer import run_genetic_experiment, compare_with_original_knn

def data_treatment():
    print("=== TRATAMENTO DOS DADOS ===")
    
    df = pd.read_csv('fetal_health.csv')
    print(f"Dataset carregado: {df.shape}")
    
    print("\nDados nulos:")
    print(df.isnull().sum())
    
    print(f"\nDados duplicados antes da remoção: {df.duplicated().sum()}")
    
    df = df.drop_duplicates()
    print(f"Dados duplicados após remoção: {df.duplicated().sum()}")
    print(f"Shape após limpeza: {df.shape}")
    
    df = df.rename(columns={'fetal_health': 'target'})
    
    print(f"\nShape (linhas, colunas): {df.shape}")
    print(f"\nTipos de dados:\n{df.dtypes}")
    
    print(f"\nInformações do dataset:")
    df.info()
    
    return df

def data_preprocessing(df):
    print("\n=== PRÉ-PROCESSAMENTO DOS DADOS ===")
    
    X = df.drop("target", axis=1)
    y = df["target"]
    
    print(f"Features (X): {X.shape}")
    print(f"Target (y): {y.shape}")
    
    x_columns = X.columns.tolist()
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), x_columns)
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    print(f"\nSplit dos dados:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"\nApós SMOTE:")
    print(f"X_resampled: {X_resampled.shape}, y_resampled: {y_resampled.shape}")
    
    contagem = y_resampled.value_counts().sort_index()
    
    plt.figure(figsize=(8, 5))
    plt.bar(contagem.index.astype(str), contagem.values)
    plt.title('Distribuição das Classes após SMOTE')
    plt.xlabel('Classes')
    plt.ylabel('Quantidade')
    plt.tight_layout()
    plt.show()
    
    return X, y, X_train, X_test, y_train, y_test, X_resampled, y_resampled, preprocessor

def data_interpretation(df):
    print("\n=== INTERPRETAÇÃO DOS DADOS ===")
    
    df['target'].value_counts().plot(kind='bar')
    plt.title("Distribuição da variável Target")
    plt.xlabel("Classificação")
    plt.ylabel("Quantidade")
    plt.show()
    
    print("A partir da distribuição da variável target, é possível perceber que esse dataset é desbalanceado.")
    print("A maior parte dos dados são classificados como Normal.")
    
    correlation_matrix = df.corr()
    
    plt.figure(figsize=(16, 16))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de correlação entre variáveis')
    plt.tight_layout()
    plt.show()
    
    correlation_cols = [
        'abnormal_short_term_variability',
        'mean_value_of_short_term_variability',
        'prolongued_decelerations',
        'histogram_mean',
        'histogram_median',
        'target'
    ]
    
    sns.pairplot(df[correlation_cols], hue='target', diag_kind='hist')
    plt.suptitle("Pairplot das Variáveis Mais Correlacionadas")
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='target', y='abnormal_short_term_variability')
    plt.title("Boxplot - Abnormal Short Term Variability")
    plt.grid(True)
    plt.show()
    
    print("\nInterpretação abnormal_short_term_variability:")
    print("Valores da abnormal_short_term_variability e a distribuição geral aumentam conforme o target vai de 1 → 2 → 3.")
    print("Isso sugere que maior variabilidade anormal de curto prazo está associada a maior risco fetal.")
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='target', y='prolongued_decelerations')
    plt.title("Boxplot - Prolongued Decelerations")
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='prolongued_decelerations', hue='target')
    plt.title("Contagem de valores de prolongued_decelerations por target")
    plt.grid(True)
    plt.show()
    
    print("\nInterpretação prolongued_decelerations:")
    print("A maioria dos valores é zero, especialmente nas classes 1 e 2.")
    print("A classe 3 apresenta uma maior frequência de valores diferentes de zero.")
    print("A classe 3 tende a apresentar maior incidência de desacelerações prolongadas,")
    print("o que faz sentido do ponto de vista clínico: desacelerações prolongadas estão associadas a sofrimento fetal.")

def knn_configuration(X_resampled, y_resampled, X_test, y_test):
    print("\n=== CONFIGURAÇÃO DO KNN ===")
    
    error = []
    
    for i in range(1, 10):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_resampled, y_resampled)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 10), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    plt.grid(True)
    plt.show()
    
    best_k = np.argmin(error) + 1
    print(f"Melhor valor de K encontrado: {best_k}")
    print(f"Erro mínimo: {min(error):.4f}")
    
    return best_k

def train_test_knn(X_resampled, y_resampled, X_test, y_test, preprocessor, best_k):
    print("\n=== TREINO E TESTE DO MODELO KNN ===")
    
    knn_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=best_k))
    ])
    
    knn_pipeline.fit(X_resampled, y_resampled)
    print("Modelo treinado com sucesso!")
    
    y_pred_knn = knn_pipeline.predict(X_test)
    print("Predições realizadas no conjunto de teste!")
    
    return knn_pipeline, y_pred_knn

def show_results(y_test, y_pred_knn, knn_pipeline, X_test, X):
    print("\n=== RESULTADOS DO MODELO KNN ===")
    
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    recall_knn = recall_score(y_test, y_pred_knn, average='macro')
    f1_knn = f1_score(y_test, y_pred_knn, average='macro')
    
    print(f"Acurácia com KNN: {accuracy_knn:.4f}")
    print(f"Taxa de verdadeiro positivo (Recall) com KNN: {recall_knn:.4f}")
    print(f"F1-score com KNN: {f1_knn:.4f}")
    
    knn_report = classification_report(y_test, y_pred_knn)
    print(f"\nRelatório de Classificação KNN:\n{knn_report}")
    
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    labels_confusion_matrix = ['Normal', 'Suspect', 'Pathological']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_confusion_matrix, yticklabels=labels_confusion_matrix)
    plt.xlabel('Classe Predita')
    plt.ylabel('Classe Real')
    plt.title('Matriz de Confusão - KNN')
    plt.show()
    
    print("\nGerando análise SHAP...")
    X_test_transformed_knn = knn_pipeline.named_steps['preprocessor'].transform(X_test)
    model_knn = knn_pipeline.named_steps['classifier']
    predict_fn = lambda x: model_knn.predict_proba(x)
    
    background = shap.sample(X_test_transformed_knn, 100)
    explainer_knn = shap.KernelExplainer(predict_fn, background)
    
    shap_values_knn = explainer_knn.shap_values(X_test_transformed_knn, nsamples=100)
    
    plt.figure()
    shap.summary_plot(
        shap_values_knn[1] if isinstance(shap_values_knn, list) else shap_values_knn,
        features=X_test_transformed_knn,
        feature_names=X.columns,
        show=False
    )
    plt.title('SHAP Summary Plot - KNN')
    plt.tight_layout()
    plt.show()
    
    print("Análise SHAP concluída!")

def genetic_optimization_experiments(X_train, y_train, X_test, y_test, preprocessor, best_k):
    print("\n=== OTIMIZAÇÃO GENÉTICA DE HIPERPARÂMETROS KNN ===")
    
    experiments = []
    
    experiment1 = run_genetic_experiment(
        X_train, y_train, X_test, y_test, preprocessor,
        "Experimento 1 - Configuração Conservadora",
        population_size=20,
        generations=15,
        mutation_rate=0.05,
        crossover_rate=0.7
    )
    experiments.append(experiment1)
    
    experiment2 = run_genetic_experiment(
        X_train, y_train, X_test, y_test, preprocessor,
        "Experimento 2 - Configuração Balanceada",
        population_size=30,
        generations=20,
        mutation_rate=0.1,
        crossover_rate=0.8
    )
    experiments.append(experiment2)
    
    experiment3 = run_genetic_experiment(
        X_train, y_train, X_test, y_test, preprocessor,
        "Experimento 3 - Configuração Agressiva",
        population_size=50,
        generations=25,
        mutation_rate=0.15,
        crossover_rate=0.9
    )
    experiments.append(experiment3)
    
    best_experiment = max(experiments, key=lambda x: x['best_fitness'])
    
    print(f"\n{'='*80}")
    print("RESUMO DOS EXPERIMENTOS")
    print(f"{'='*80}")
    
    for exp in experiments:
        print(f"\n{exp['experiment_name']}:")
        print(f"  Melhor Fitness: {exp['best_fitness']:.4f}")
        print(f"  Accuracy: {exp['best_metrics']['accuracy']:.4f}")
        print(f"  Recall: {exp['best_metrics']['recall']:.4f}")
        print(f"  F1-Score: {exp['best_metrics']['f1_score']:.4f}")
    
    print(f"\n🏆 MELHOR EXPERIMENTO: {best_experiment['experiment_name']}")
    print(f"Melhor Fitness: {best_experiment['best_fitness']:.4f}")
    print(f"Melhores Parâmetros: {best_experiment['best_params']}")
    
    comparison = compare_with_original_knn(
        X_train, y_train, X_test, y_test, preprocessor,
        best_experiment['best_params'], best_k
    )
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(experiment1['fitness_history']) + 1), experiment1['fitness_history'], 'b-', label='Experimento 1')
    plt.plot(range(1, len(experiment2['fitness_history']) + 1), experiment2['fitness_history'], 'r-', label='Experimento 2')
    plt.plot(range(1, len(experiment3['fitness_history']) + 1), experiment3['fitness_history'], 'g-', label='Experimento 3')
    plt.title('Evolução do Fitness - Todos os Experimentos')
    plt.xlabel('Geração')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    metrics = ['Accuracy', 'Recall', 'F1-Score']
    original_values = [comparison['original']['accuracy'], comparison['original']['recall'], comparison['original']['f1_score']]
    optimized_values = [comparison['optimized']['accuracy'], comparison['optimized']['recall'], comparison['optimized']['f1_score']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, original_values, width, label='Original', alpha=0.8)
    plt.bar(x + width/2, optimized_values, width, label='Otimizado', alpha=0.8)
    plt.title('Comparação de Métricas')
    plt.xlabel('Métricas')
    plt.ylabel('Valor')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    improvements = [
        (comparison['optimized']['accuracy'] - comparison['original']['accuracy']) / comparison['original']['accuracy'] * 100,
        (comparison['optimized']['recall'] - comparison['original']['recall']) / comparison['original']['recall'] * 100,
        (comparison['optimized']['f1_score'] - comparison['original']['f1_score']) / comparison['original']['f1_score'] * 100
    ]
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    plt.bar(metrics, improvements, color=colors, alpha=0.7)
    plt.title('Melhorias (%)')
    plt.xlabel('Métricas')
    plt.ylabel('Melhoria (%)')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return experiments, best_experiment, comparison

def main():
    print("=== ANÁLISE DE SAÚDE FETAL COM KNN ===\n")
    
    df = data_treatment()
    
    data_interpretation(df)
    
    X, y, X_train, X_test, y_train, y_test, X_resampled, y_resampled, preprocessor = data_preprocessing(df)
    
    best_k = knn_configuration(X_resampled, y_resampled, X_test, y_test)
    
    knn_pipeline, y_pred_knn = train_test_knn(X_resampled, y_resampled, X_test, y_test, preprocessor, best_k)
    
    show_results(y_test, y_pred_knn, knn_pipeline, X_test, X)
    
    experiments, best_experiment, comparison = genetic_optimization_experiments(
        X_resampled, y_resampled, X_test, y_test, preprocessor, best_k
    )
    
    print("\n=== ANÁLISE CONCLUÍDA ===")

if __name__ == "__main__":
    main()