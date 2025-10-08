import json
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from joblib import load
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from streamlit_chat import message
import os

api_key = ""
client = ChatOpenAI(model="gpt-5", openai_api_key=api_key)

# Caminho absoluto para o modelo
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
modelo_path = os.path.join(base_dir, "data", "modelo_classificacaofetal01_20251007_202709.joblib")
modelo = load(modelo_path)

colunas = [
    "baseline value", "accelerations", "fetal_movement", "uterine_contractions",
    "light_decelerations", "severe_decelerations", "prolongued_decelerations",
    "abnormal_short_term_variability", "mean_value_of_short_term_variability",
    "percentage_of_time_with_abnormal_long_term_variability", "mean_value_of_long_term_variability",
    "histogram_width", "histogram_min", "histogram_max",
    "histogram_number_of_peaks", "histogram_number_of_zeroes", "histogram_mode",
    "histogram_mean", "histogram_median", "histogram_variance", "histogram_tendency"
]

# Valores de exemplo baseados nos dados reais para inicialização dos inputs
valores_exemplo = {
    "baseline value": 132.0,
    "accelerations": 0.003,
    "fetal_movement": 0.0,
    "uterine_contractions": 0.008,
    "light_decelerations": 0.003,
    "severe_decelerations": 0.0,
    "prolongued_decelerations": 0.0,
    "abnormal_short_term_variability": 16.0,
    "mean_value_of_short_term_variability": 2.1,
    "percentage_of_time_with_abnormal_long_term_variability": 0.0,
    "mean_value_of_long_term_variability": 10.4,
    "histogram_width": 130.0,
    "histogram_min": 68.0,
    "histogram_max": 198.0,
    "histogram_number_of_peaks": 6.0,
    "histogram_number_of_zeroes": 1.0,
    "histogram_mode": 141.0,
    "histogram_mean": 136.0,
    "histogram_median": 140.0,
    "histogram_variance": 12.0,
    "histogram_tendency": 0.0
}

def interpretar(features, resultado):
    """
    Usa prompt engineering para pedir à LLM interpretação médica
    e insights acionáveis com base nos resultados do modelo.
    """
    try:
        medical_prompt = """
          Você é um assistente médico especializado em saúde fetal.
          Você deve interpretar resultados de cardiotocografia fetal e gerar um relatório clínico.

          Dados das medições: {features}
          Resultado da predição: {resultado} (1: Normal, 2: Suspeito, 3: Patológico)

          Por favor, gere:
          1. Uma explicação clara em linguagem natural.
          2. Possíveis causas.
          3. Recomendações médicas.
          4. Um nível de urgência.
          5. Aviso de que a classificação pode cometer erros e deve ser avaliada por um profissional.

          Responda em formato estruturado JSON com campos:
          - explicacao
          - causas
          - recomendacoes
          - urgencia
          - aviso
          """

        prompt_template = ChatPromptTemplate.from_template(medical_prompt)
        prompt = prompt_template.format(features=json.dumps(features), resultado=resultado)

        # Use invoke method instead of chat
        response = client.invoke(prompt)

        return response.content
    except Exception as e:
        error_msg = f"Erro ao interpretar resultado com LLM: {str(e)}"
        st.error(error_msg)
        raise Exception(error_msg) from e

def gerar_valores_aleatorios():
    """
    Gera valores aleatórios baseados nas estatísticas dos dados reais.
    """
    return {
        "baseline value": np.random.normal(132.0, 15.0),
        "accelerations": np.random.exponential(0.005),
        "fetal_movement": np.random.choice([0.0, 0.001, 0.002]),
        "uterine_contractions": np.random.exponential(0.008),
        "light_decelerations": np.random.exponential(0.003),
        "severe_decelerations": np.random.choice([0.0, 0.001, 0.002]),
        "prolongued_decelerations": np.random.choice([0.0, 0.001]),
        "abnormal_short_term_variability": np.random.normal(20.0, 15.0),
        "mean_value_of_short_term_variability": np.random.normal(2.5, 1.0),
        "percentage_of_time_with_abnormal_long_term_variability": np.random.exponential(5.0),
        "mean_value_of_long_term_variability": np.random.normal(15.0, 8.0),
        "histogram_width": np.random.normal(120.0, 20.0),
        "histogram_min": np.random.normal(70.0, 15.0),
        "histogram_max": np.random.normal(180.0, 25.0),
        "histogram_number_of_peaks": np.random.poisson(5.0),
        "histogram_number_of_zeroes": np.random.poisson(1.0),
        "histogram_mode": np.random.normal(135.0, 15.0),
        "histogram_mean": np.random.normal(135.0, 15.0),
        "histogram_median": np.random.normal(135.0, 15.0),
        "histogram_variance": np.random.exponential(15.0),
        "histogram_tendency": np.random.choice([-1.0, 0.0, 1.0])
    }

def predizer(features):
    """
    Realiza predição usando DataFrame para melhor estruturação dos dados.

    Args:
        features: Lista de valores ou dicionário com as features

    Returns:
        int: Resultado da predição (1: Normal, 2: Suspeito, 3: Patológico)
    """
    try:
        # Converte features para DataFrame
        if isinstance(features, list):
            # Se for lista, cria DataFrame com as colunas definidas
            df_features = pd.DataFrame([features], columns=colunas)
        elif isinstance(features, dict):
            # Se for dicionário, cria DataFrame diretamente
            df_features = pd.DataFrame([features])
        else:
            raise ValueError("Features deve ser uma lista ou dicionário")

        # Realiza a predição
        prediction = modelo.predict(df_features)

        return prediction[0]
    except Exception as e:
        error_msg = f"Erro ao realizar predição do modelo: {str(e)}"
        st.error(error_msg)
        raise Exception(error_msg) from e

def get_response_from_model(features):
    """
    Processa features e retorna interpretação médica.

    Args:
        features: Lista de valores ou dicionário com as features

    Returns:
        str: Interpretação médica da predição
    """
    try:
        resultado = predizer(features)

        # Converte features para dicionário para melhor interpretação
        if isinstance(features, list):
            features_dict = dict(zip(colunas, features))
        else:
            features_dict = features

        explicacao = interpretar(features_dict, resultado)
        salvar_log(features_dict, resultado, explicacao)
        return explicacao
    except Exception as e:
        error_msg = f"Erro ao processar a predição: {str(e)}"
        st.error(error_msg)
        raise

def salvar_log(features, resultado, explicacao):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = {
        "timestamp": timestamp,
        "features": features,
        "resultado": resultado,
        "explicacao": explicacao
    }
    with open("log_interpretacoes.jsonl", "a") as f:
        f.write(json.dumps(log) + "\n")

def on_input_change():
    user_input = st.session_state.user_input
    response_text = get_response_from_model(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append({"type": "normal", "data": response_text})

def on_btn_click():
    del st.session_state.past[:]
    del st.session_state.generated[:]

st.session_state.setdefault('past', [])
st.session_state.setdefault('generated', [])

st.title("Chat de Avaliação da Saúde Fetal")

chat_placeholder = st.empty()

with chat_placeholder.container():
    for i in range(len(st.session_state['generated'])):
        message(st.session_state['past'][i], is_user=True, key=f"{i}_user")
        message(
            st.session_state['generated'][i]['data'],
            key=f"{i}",
            allow_html=True,
            is_table=False
        )

    st.button("Limpar histórico", on_click=on_btn_click)

with st.form("input_form"):
    st.write("Preencha os dados da cardiotocografia:")
    
    # Botão para gerar valores aleatórios
    col1, col2 = st.columns(2)
    with col1:
        if st.form_submit_button("🎲 Gerar Valores Aleatórios", type="secondary"):
            valores_aleatorios = gerar_valores_aleatorios()
            st.session_state.valores_aleatorios = valores_aleatorios

    # Inicializa valores aleatórios se não existirem
    if 'valores_aleatorios' not in st.session_state:
        st.session_state.valores_aleatorios = valores_exemplo

    entradas = {}
    for col in colunas:
        valor_atual = st.session_state.valores_aleatorios.get(col, valores_exemplo[col])
        entradas[col] = st.number_input(col, value=valor_atual)

    submitted = st.form_submit_button("Realizar Predição")

    if submitted:
        try:
            valores = [entradas[col] for col in colunas]
            explicacao = get_response_from_model(valores)

            st.session_state.past.append(valores)
            st.session_state.generated.append({"type": "normal", "data": explicacao})
            st.rerun()
        except Exception as e:
            st.error(f"❌ Erro ao realizar predição: {str(e)}")
            st.exception(e)