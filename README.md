# Tech Challenge - Classificação de Saúde Fetal

Projeto para classificação da saúde fetal usando algoritmo KNN otimizado com algoritmo genético e interface de chat com IA para interpretação dos resultados.

## Descrição

O sistema utiliza dados de cardiotocografia para classificar a saúde fetal em três categorias:

- **Normal**: Parâmetros dentro da normalidade
- **Suspeito**: Alguns parâmetros requerem atenção
- **Patológico**: Necessidade de intervenção

O modelo KNN é otimizado através de algoritmo genético para melhor performance, e os resultados podem ser interpretados por uma interface de chat com LLM.

## Como Executar

### 1. Instalação de Dependências

```bash
pip install -r requirements.txt
```

### 2. Treinar e Otimizar o Modelo

```bash
python app.py
```

Este comando:

- Treina o modelo KNN original
- Executa 3 experimentos de otimização com algoritmo genético
- Salva o melhor modelo em `data/`
- Exibe comparação de métricas entre modelo original e otimizado

### 3. Executar o Chat de Análise

```bash
python -m streamlit run chat/chat_llm.py
```

**Importante:** Configure sua chave da API OpenAI no arquivo `chat/chat_llm.py` (linha 12):

```python
api_key = "sua-chave-aqui"
```

A interface permite:

- Inserir manualmente os dados de cardiotocografia
- Gerar valores aleatórios para testes
- Obter predição do modelo
- Receber interpretação médica detalhada via LLM

## Estrutura do Projeto

```
├── app.py                          # Script principal para treinamento
├── fetal_health_KNN.py            # Implementação do KNN
├── genetic_algorithm_knn_optmizer.py  # Otimização com algoritmo genético
├── chat/
│   └── chat_llm.py                # Interface Streamlit com chat IA
├── data/
│   ├── fetal_health.csv           # Dataset
│   └── modelo_*.joblib            # Modelos salvos
└── requirements.txt               # Dependências
```

## Equipe

- **Bruna Cardoso Andrade** - RM366295
- **Felipe de Siqueira Zanella** - RM365834
- **Vinicius de Souza Medeiros** - RM366459
- **Horacy Lopes da Silva Junior** - RM365525
- **Gabriel Luiz Santana** - RM366287

---
