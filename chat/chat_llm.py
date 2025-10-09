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
    "fetal_movement": 0.01,
    "uterine_contractions": 0.008,
    "light_decelerations": 0.003,
    "severe_decelerations": 0.01,
    "prolongued_decelerations": 0.01,
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

def interpretar(features, resultado, stream_placeholder=None):
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

        # Mostra indicador de carregamento se temos placeholder
        if stream_placeholder:
            stream_placeholder.info("🤖 **Aguardando resposta da IA...**")

        # Usa invoke sem streaming (streaming requer verificação da organização)
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

def formatar_resposta(resultado, explicacao_json):
    """
    Formata a resposta de forma legível e estruturada seguindo boas práticas de UX.

    Args:
        resultado: int - Resultado da predição (1: Normal, 2: Suspeito, 3: Patológico)
        explicacao_json: str - JSON com a interpretação médica

    Returns:
        dict: Dicionário com dados formatados
    """
    # Mapeamento de resultados
    resultado_map = {
        1: {
            "status": "Normal",
            "emoji": "🟢",
            "cor": "success",
            "descricao": "Os parâmetros fetais estão dentro da normalidade"
        },
        2: {
            "status": "Suspeito",
            "emoji": "🟡",
            "cor": "warning",
            "descricao": "Alguns parâmetros fetais requerem atenção"
        },
        3: {
            "status": "Patológico",
            "emoji": "🔴",
            "cor": "error",
            "descricao": "Parâmetros fetais indicam necessidade de intervenção"
        }
    }

    info_resultado = resultado_map.get(resultado, {
        "status": "Desconhecido",
        "emoji": "⚪",
        "cor": "info",
        "descricao": "Resultado não identificado"
    })

    try:
        # Tenta fazer parse do JSON
        explicacao = json.loads(explicacao_json)
    except:
        # Se não for JSON válido, cria estrutura básica
        explicacao = {
            "explicacao": explicacao_json,
            "causas": "Não disponível",
            "recomendacoes": "Consulte um profissional de saúde",
            "urgencia": "Indeterminada",
            "aviso": "Esta análise é automatizada e deve ser avaliada por um profissional de saúde."
        }

    return {
        "resultado": info_resultado,
        "explicacao": explicacao
    }

def get_response_from_model(features, stream_placeholder=None):
    """
    Processa features e retorna interpretação médica.

    Args:
        features: Lista de valores ou dicionário com as features
        stream_placeholder: Container para streaming da resposta

    Returns:
        tuple: (resultado, explicacao_json)
    """
    try:
        resultado = predizer(features)

        # Converte features para dicionário para melhor interpretação
        if isinstance(features, list):
            features_dict = dict(zip(colunas, features))
        else:
            features_dict = features

        explicacao_json = interpretar(features_dict, resultado, stream_placeholder)

        salvar_log(features_dict, resultado, explicacao_json)
        return resultado, explicacao_json
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

st.session_state.setdefault('historico', [])

st.title("🏥 Chat de Avaliação da Saúde Fetal")
st.markdown("---")

with st.form("input_form"):
    st.subheader("📋 Dados da Cardiotocografia")

    # Botão para gerar valores aleatórios
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.form_submit_button("🎲 Gerar Valores Aleatórios", type="secondary"):
            valores_aleatorios = gerar_valores_aleatorios()
            st.session_state.valores_aleatorios = valores_aleatorios
            st.rerun()

    # Inicializa valores aleatórios se não existirem
    if 'valores_aleatorios' not in st.session_state:
        st.session_state.valores_aleatorios = valores_exemplo

    st.markdown("---")

    # Organiza inputs em colunas para melhor visualização
    col1, col2 = st.columns(2)
    entradas = {}

    for idx, col in enumerate(colunas):
        valor_atual = st.session_state.valores_aleatorios.get(col, valores_exemplo[col])
        if idx % 2 == 0:
            with col1:
                entradas[col] = st.number_input(col, value=valor_atual, key=f"input_{col}")
        else:
            with col2:
                entradas[col] = st.number_input(col, value=valor_atual, key=f"input_{col}")

    st.markdown("---")
    submitted = st.form_submit_button("🔬 Realizar Predição", use_container_width=True, type="primary")

# Área de resultados (fora do formulário)
st.markdown("---")
st.subheader("📊 Resultados da Análise")

if submitted:
    try:
        valores = [entradas[col] for col in colunas]

        # Indicadores de progresso
        progress_container = st.container()
        with progress_container:
            status_text = st.empty()
            status_text.info("📊 Preparando dados para análise...")

            status_text.info("🤖 Executando modelo de predição...")

            # Container para streaming
            status_text.info("🧠 Gerando interpretação médica com IA...")
            stream_container = st.empty()

            # Executa a predição com streaming
            resultado, explicacao_json = get_response_from_model(valores, stream_container)

            # Limpa o status
            status_text.empty()
            stream_container.empty()

        # Formata a resposta
        dados_formatados = formatar_resposta(resultado, explicacao_json)
        info_resultado = dados_formatados['resultado']
        explicacao = dados_formatados['explicacao']

        # Exibe resultado formatado
        st.success("✅ Análise concluída com sucesso!")

        # Cabeçalho do resultado
        st.markdown(f"## {info_resultado['emoji']} Diagnóstico: **{info_resultado['status']}**")
        st.caption(info_resultado['descricao'])

        st.markdown("---")

        # Explicação
        with st.container():
            st.markdown("### 📋 Explicação do Resultado")
            st.write(explicacao.get('explicacao', 'Não disponível'))

        st.markdown("")

        # Possíveis Causas
        with st.container():
            st.markdown("### 🔍 Possíveis Causas")
            st.write(explicacao.get('causas', 'Não disponível'))

        st.markdown("")

        # Recomendações
        with st.container():
            st.markdown("### 💊 Recomendações Médicas")
            st.write(explicacao.get('recomendacoes', 'Não disponível'))

        st.markdown("")

        # Nível de Urgência
        urgencia = explicacao.get('urgencia', 'Não disponível')
        with st.container():
            st.markdown("### ⚡ Nível de Urgência")

            # Define a cor baseada no nível de urgência
            if any(palavra in urgencia.lower() for palavra in ['alta', 'urgente', 'imediata', 'crítica']):
                st.error(f"**{urgencia}**")
            elif any(palavra in urgencia.lower() for palavra in ['média', 'moderada', 'atenção']):
                st.warning(f"**{urgencia}**")
            else:
                st.info(f"**{urgencia}**")

        st.markdown("")

        # Aviso
        with st.container():
            st.warning(f"⚠️ **Aviso Importante**\n\n{explicacao.get('aviso', 'Esta análise é automatizada e deve ser avaliada por um profissional de saúde.')}")

        # Salva no histórico
        st.session_state.historico.append({
            'valores': valores,
            'resultado': resultado,
            'dados_formatados': dados_formatados
        })

    except Exception as e:
        st.error(f"❌ Erro ao realizar predição: {str(e)}")
        with st.expander("📋 Detalhes do erro"):
            st.exception(e)

# Mostra histórico se existir
elif len(st.session_state.historico) > 0:
    ultimo_item = st.session_state.historico[-1]
    dados = ultimo_item['dados_formatados']
    info_resultado = dados['resultado']
    explicacao = dados['explicacao']

    # Exibe o último resultado
    st.markdown(f"## {info_resultado['emoji']} Diagnóstico: **{info_resultado['status']}**")
    st.caption(info_resultado['descricao'])
    st.markdown("---")

    with st.container():
        st.markdown("### 📋 Explicação do Resultado")
        st.write(explicacao.get('explicacao', 'Não disponível'))
    st.markdown("")

    with st.container():
        st.markdown("### 🔍 Possíveis Causas")
        st.write(explicacao.get('causas', 'Não disponível'))
    st.markdown("")

    with st.container():
        st.markdown("### 💊 Recomendações Médicas")
        st.write(explicacao.get('recomendacoes', 'Não disponível'))
    st.markdown("")

    urgencia = explicacao.get('urgencia', 'Não disponível')
    with st.container():
        st.markdown("### ⚡ Nível de Urgência")
        if any(palavra in urgencia.lower() for palavra in ['alta', 'urgente', 'imediata', 'crítica']):
            st.error(f"**{urgencia}**")
        elif any(palavra in urgencia.lower() for palavra in ['média', 'moderada', 'atenção']):
            st.warning(f"**{urgencia}**")
        else:
            st.info(f"**{urgencia}**")
    st.markdown("")

    with st.container():
        st.warning(f"⚠️ **Aviso Importante**\n\n{explicacao.get('aviso', 'Esta análise é automatizada e deve ser avaliada por um profissional de saúde.')}")

    # Mostra histórico anterior
    if len(st.session_state.historico) > 1:
        st.markdown("---")
        with st.expander(f"📜 Ver histórico ({len(st.session_state.historico) - 1} análises anteriores)"):
            for i, item in enumerate(reversed(st.session_state.historico[:-1])):
                dados_hist = item['dados_formatados']
                info_hist = dados_hist['resultado']
                exp_hist = dados_hist['explicacao']

                st.markdown(f"#### Análise #{len(st.session_state.historico) - i - 1} - {info_hist['emoji']} {info_hist['status']}")
                st.write(f"**Explicação:** {exp_hist.get('explicacao', 'Não disponível')}")
                st.write(f"**Urgência:** {exp_hist.get('urgencia', 'Não disponível')}")
                st.markdown("---")
else:
    st.info("👆 Preencha os dados acima e clique em 'Realizar Predição' para começar a análise.")