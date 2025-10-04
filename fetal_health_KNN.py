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

##Código original do algoritmo de KNN do tech_challeng_classificao_saude_fetal separado por funções.
##Foi removido a parte de interpretação dos dados e dos gráficos.

class FetalHealthKNN:
    def data_treatment():
        df = pd.read_csv('data/fetal_health.csv')
        df = df.drop_duplicates()        
        df = df.rename(columns={'fetal_health': 'target'})
        return df

    def data_preprocessing(df):    
        X = df.drop("target", axis=1)
        y = df["target"]
        
        x_columns = X.columns.tolist()
        
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), x_columns)
        ])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
             
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
        
        return best_k

    def train_test_knn(X_resampled, y_resampled, X_test, y_test, preprocessor, best_k):    
        knn_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', KNeighborsClassifier(n_neighbors=best_k))
        ])
        
        knn_pipeline.fit(X_resampled, y_resampled)    
        y_pred_knn = knn_pipeline.predict(X_test)
        
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

    def run_fetal_health_knn():       
        df = FetalHealthKNN.data_treatment()
         
        X, y, X_train, X_test, y_train, y_test, X_resampled, y_resampled, preprocessor = FetalHealthKNN.data_preprocessing(df)
        
        best_k = FetalHealthKNN.knn_configuration(X_resampled, y_resampled, X_test, y_test)
        
        knn_pipeline, y_pred_knn = FetalHealthKNN.train_test_knn(X_resampled, y_resampled, X_test, y_test, preprocessor, best_k)
        
        FetalHealthKNN.show_results(y_test, y_pred_knn, knn_pipeline, X_test, X)
        
        return X_train, y_train, X_test, y_test, preprocessor, X_resampled, y_resampled