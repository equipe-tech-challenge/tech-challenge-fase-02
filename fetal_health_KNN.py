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


def data_treatment():
    df = pd.read_csv('data/fetal_health.csv')
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
    
    return X, y, X_train, X_test, y_train, y_test, X_resampled, y_resampled, preprocessor

def knn_configuration(X_resampled, y_resampled, X_test, y_test):    
    error = []
    
    for i in range(1, 10):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_resampled, y_resampled)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))
        
    best_k = np.argmin(error) + 1
    print(f"Melhor valor de K encontrado: {best_k}")
    print(f"Erro mínimo: {min(error):.4f}")
    
    return best_k

def train_test_knn(X_resampled, y_resampled, X_test, y_test, preprocessor, best_k):    
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
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    recall_knn = recall_score(y_test, y_pred_knn, average='macro')
    f1_knn = f1_score(y_test, y_pred_knn, average='macro')
    
    print(f"Acurácia com KNN: {accuracy_knn:.4f}")
    print(f"Taxa de verdadeiro positivo (Recall) com KNN: {recall_knn:.4f}")
    print(f"F1-score com KNN: {f1_knn:.4f}")
    
    knn_report = classification_report(y_test, y_pred_knn)
    print(f"\nRelatório de Classificação KNN:\n{knn_report}")


def main():
    print("=== ANÁLISE DE SAÚDE FETAL COM KNN ===\n")
    
    df = data_treatment()
        
    X, y, X_train, X_test, y_train, y_test, X_resampled, y_resampled, preprocessor = data_preprocessing(df)
    
    best_k = knn_configuration(X_resampled, y_resampled, X_test, y_test)
    
    knn_pipeline, y_pred_knn = train_test_knn(X_resampled, y_resampled, X_test, y_test, preprocessor, best_k)
    
    show_results(y_test, y_pred_knn, knn_pipeline, X_test, X)
    

if __name__ == "__main__":
    main()