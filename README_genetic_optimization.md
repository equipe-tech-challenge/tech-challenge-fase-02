# Otimização Genética de Hiperparâmetros KNN

## Visão Geral

Este projeto implementa um algoritmo genético para otimizar os hiperparâmetros do modelo KNN (K-Nearest Neighbors) aplicado ao diagnóstico de saúde fetal. A otimização genética permite encontrar automaticamente a melhor combinação de parâmetros para maximizar o desempenho do modelo.

## Arquivos Implementados

### 1. `genetic_knn_optimizer.py`

Contém a implementação completa do algoritmo genético para otimização de hiperparâmetros KNN.

### 2. `app.py` (atualizado)

Integra a otimização genética ao pipeline principal de análise.

## Codificação dos Hiperparâmetros

### Representação Genética

Cada indivíduo (solução) é representado como um dicionário contendo:

```python
{
    'n_neighbors': int,        # 1-20 vizinhos
    'weights': str,            # 'uniform' ou 'distance'
    'metric': str,             # 'euclidean', 'manhattan', 'minkowski'
    'algorithm': str,          # 'auto', 'ball_tree', 'kd_tree', 'brute'
    'leaf_size': int          # 10-50
}
```

### Espaço de Busca

- **n_neighbors**: 1 a 20 (inteiro)
- **weights**: 'uniform' ou 'distance' (categórico)
- **metric**: 'euclidean', 'manhattan', 'minkowski' (categórico)
- **algorithm**: 'auto', 'ball_tree', 'kd_tree', 'brute' (categórico)
- **leaf_size**: 10 a 50 (inteiro)

## Função Fitness

A função fitness combina múltiplas métricas de desempenho:

```python
fitness = (accuracy × 0.4) + (recall × 0.3) + (f1_score × 0.3)
```

**Pesos:**

- Accuracy: 40% (precisão geral)
- Recall: 30% (sensibilidade)
- F1-Score: 30% (harmonia entre precisão e recall)

## Operadores Genéticos

### 1. Seleção por Torneio

- Seleciona k=3 indivíduos aleatoriamente
- Retorna o melhor entre eles
- Promove diversidade genética

### 2. Crossover (Cruzamento)

- Taxa de crossover configurável (padrão: 80%)
- Crossover uniforme para cada parâmetro
- Cada parâmetro tem 50% de chance de ser trocado

### 3. Mutação

- Taxa de mutação configurável (padrão: 10%)
- Mutação independente para cada parâmetro
- Cada parâmetro tem 20% de chance de mutar

### 4. Elitismo

- Preserva os melhores indivíduos entre gerações
- Tamanho da elite configurável (padrão: 2)

## Experimentos Realizados

### Experimento 1: Configuração Conservadora

- **População**: 20 indivíduos
- **Gerações**: 15
- **Taxa de Mutação**: 5%
- **Taxa de Crossover**: 70%
- **Objetivo**: Exploração inicial com baixo risco

### Experimento 2: Configuração Balanceada

- **População**: 30 indivíduos
- **Gerações**: 20
- **Taxa de Mutação**: 10%
- **Taxa de Crossover**: 80%
- **Objetivo**: Equilíbrio entre exploração e explotação

### Experimento 3: Configuração Agressiva

- **População**: 50 indivíduos
- **Gerações**: 25
- **Taxa de Mutação**: 15%
- **Taxa de Crossover**: 90%
- **Objetivo**: Exploração intensiva do espaço de busca

## Métricas de Avaliação

### Métricas Primárias

- **Accuracy**: Proporção de predições corretas
- **Recall**: Sensibilidade (taxa de verdadeiros positivos)
- **F1-Score**: Média harmônica entre precisão e recall

### Métricas Secundárias

- **Fitness**: Combinação ponderada das métricas primárias
- **Convergência**: Evolução do fitness ao longo das gerações
- **Diversidade**: Variabilidade dos parâmetros na população

## Visualizações Geradas

### 1. Evolução do Fitness

- Gráfico de linha mostrando a evolução do melhor fitness por geração
- Comparação entre os três experimentos

### 2. Comparação de Métricas

- Gráfico de barras comparando modelo original vs otimizado
- Métricas: Accuracy, Recall, F1-Score

### 3. Melhorias Percentuais

- Gráfico de barras mostrando as melhorias em percentual
- Verde para melhorias positivas, vermelho para pioras

## Como Executar

```bash
python app.py
```

O programa executará automaticamente:

1. Tratamento e pré-processamento dos dados
2. Configuração inicial do KNN
3. Treinamento e avaliação do modelo original
4. **Otimização genética com 3 experimentos**
5. Comparação de resultados
6. Visualizações dos resultados

## Resultados Esperados

### Saídas do Console

- Progresso de cada geração do algoritmo genético
- Melhores parâmetros encontrados por experimento
- Comparação detalhada entre modelo original e otimizado
- Melhorias percentuais em cada métrica

### Gráficos Gerados

- Evolução do fitness para cada experimento
- Comparação visual de métricas
- Análise de melhorias percentuais

## Vantagens da Otimização Genética

### 1. Busca Global

- Explora todo o espaço de hiperparâmetros
- Evita ótimos locais
- Encontra combinações não óbvias

### 2. Múltiplas Objetivos

- Combina diferentes métricas de desempenho
- Balanceia precisão, recall e F1-score
- Adaptável a diferentes necessidades

### 3. Robustez

- Funciona com diferentes tipos de parâmetros
- Tolerante a ruído nos dados
- Converge para soluções estáveis

### 4. Interpretabilidade

- Mostra evolução da busca
- Permite análise do processo de otimização
- Facilita ajustes nos parâmetros do algoritmo

## Configurações Recomendadas

### Para Datasets Pequenos (< 1000 amostras)

- População: 20-30
- Gerações: 15-20
- Taxa de mutação: 5-10%

### Para Datasets Médios (1000-10000 amostras)

- População: 30-50
- Gerações: 20-30
- Taxa de mutação: 10-15%

### Para Datasets Grandes (> 10000 amostras)

- População: 50-100
- Gerações: 25-40
- Taxa de mutação: 10-20%

## Limitações e Considerações

### 1. Tempo de Execução

- Algoritmo genético é computacionalmente intensivo
- Tempo aumenta com população e gerações
- Recomenda-se usar validação cruzada para datasets grandes

### 2. Overfitting

- Risco de otimizar para o conjunto de teste
- Recomenda-se validação cruzada ou conjunto de validação separado

### 3. Parâmetros do Algoritmo

- Taxa de mutação muito alta pode causar instabilidade
- Taxa de crossover muito baixa pode limitar exploração
- Tamanho da população deve ser balanceado com tempo disponível

## Extensões Futuras

### 1. Algoritmos Adicionais

- Otimização para SVM, Random Forest, etc.
- Comparação entre diferentes algoritmos

### 2. Métricas Avançadas

- ROC-AUC, Precision-Recall curves
- Métricas específicas para problemas multiclasse

### 3. Técnicas Avançadas

- Algoritmos genéticos multi-objetivo
- Estratégias de nicho para manter diversidade
- Algoritmos híbridos (GA + outros métodos de otimização)
