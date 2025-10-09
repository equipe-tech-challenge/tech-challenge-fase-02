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
    "baseline value": 134.0,
    "accelerations": 0.001,
    "fetal_movement": 0.0,
    "uterine_contractions": 0.01,
    "light_decelerations": 0.009,
    "severe_decelerations": 0.0,
    "prolongued_decelerations": 0.002,
    "abnormal_short_term_variability": 26.0,
    "mean_value_of_short_term_variability": 5.9,
    "percentage_of_time_with_abnormal_long_term_variability": 0.0,
    "mean_value_of_long_term_variability": 0.0,
    "histogram_width": 150.0,
    "histogram_min": 50.0,
    "histogram_max": 200.0,
    "histogram_number_of_peaks": 5.0,
    "histogram_number_of_zeroes": 3.0,
    "histogram_mode": 76.0,
    "histogram_mean": 107.0,
    "histogram_median": 107.0,
    "histogram_variance": 170.0,
    "histogram_tendency": 0.0
}

def interpretar(features, resultado, stream_placeholder=None):
    """
    Usa prompt engineering para pedir à LLM interpretação médica
    e insights acionáveis com base nos resultados do modelo.
    Implementa streaming incremental da resposta.
    """
    try:
        medical_prompt = """
          Você é um assistente médico especializado em saúde fetal.
          Você deve interpretar resultados de cardiotocografia fetal e gerar um relatório clínico.

          Dados das medições: {features}
          Resultado da predição: {resultado} (1: Normal, 2: Suspeito, 3: Patológico)

          Por favor, gere uma breve análise em texto corrido (NÃO em JSON) com:

          **DIAGNÓSTICO**
          [Explicação clara e breve sobre o resultado]

          **INSIGHT**
          [Transforma o diagnóstico em um insight acionável para o médico]
          """

        prompt_template = ChatPromptTemplate.from_template(medical_prompt)
        prompt = prompt_template.format(features=json.dumps(features), resultado=resultado)

        # Se temos placeholder, tenta usar streaming
        if stream_placeholder:
            try:
                # Tenta streaming (requer organização verificada)
                full_response = ""

                for chunk in client.stream(prompt):
                    if hasattr(chunk, 'content') and chunk.content:
                        full_response += chunk.content
                        # Atualiza em tempo real com o texto acumulado
                        stream_placeholder.markdown(full_response + " ▌")

                # Remove o cursor ao finalizar
                stream_placeholder.markdown(full_response)
                return full_response

            except Exception as stream_error:
                # Se streaming falhar (organização não verificada), usa invoke
                if "stream" in str(stream_error).lower() or "unsupported" in str(stream_error).lower():
                    stream_placeholder.info("🤖 **Aguardando resposta da IA...**")
                    response = client.invoke(prompt)
                    # Mostra a resposta completa
                    stream_placeholder.markdown(response.content)
                    return response.content
                else:
                    raise stream_error
        else:
            # Sem placeholder, usa invoke direto
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

def formatar_resposta(resultado, texto_resposta):
    """
    Formata a resposta de forma legível e estruturada seguindo boas práticas de UX.

    Args:
        resultado: int - Resultado da predição (1: Normal, 2: Suspeito, 3: Patológico)
        texto_resposta: str - Texto com a interpretação médica

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

    # Extrai seções do texto usando marcadores
    import re

    def extrair_secao(texto, inicio, fim=None):
        """Extrai uma seção do texto entre dois marcadores"""
        padrao_inicio = re.escape(inicio)
        if fim:
            padrao_fim = re.escape(fim)
            padrao = f"{padrao_inicio}(.*?)(?:{padrao_fim}|$)"
        else:
            padrao = f"{padrao_inicio}(.*?)$"

        match = re.search(padrao, texto, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    explicacao_secoes = {
        "diagnostico": extrair_secao(texto_resposta, "**DIAGNÓSTICO**", "**INSIGHT**") or
                       extrair_secao(texto_resposta, "DIAGNÓSTICO", "INSIGHT") or
                       "Análise em andamento...",
        "INSIGHT": extrair_secao(texto_resposta, "**INSIGHT**", "**RECOMENDAÇÕES MÉDICAS**") or
                    extrair_secao(texto_resposta, "INSIGHT") or
                    "Sem resultados disponíveis.",
        "aviso": "AVISO IMPORTANTE Esta é uma análise automatizada por IA e não substitui "
        "a avaliação clínica. A interpretação definitiva deve considerar idade gestacional, "
        "sinais maternos, qualidade do traçado e contexto obstétrico, sendo o diagnóstico final "
        "de responsabilidade do médico assistente. Em caso de sintomas maternos, redução de "
        "movimentos fetais ou fatores de risco, recomenda-se reavaliação clínica e,"
        " se necessário, repetição do CTG e exames complementares."
    }

    return {
        "resultado": info_resultado,
        "explicacao": explicacao_secoes,
        "texto_completo": texto_resposta
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

        # Status de preparação
        status_text = st.empty()
        status_text.info("📊 Preparando dados para análise...")

        # Executa modelo de predição
        status_text.info("🤖 Executando modelo de predição...")

        # Converte features para dicionário
        if isinstance(valores, list):
            features_dict = dict(zip(colunas, valores))
        else:
            features_dict = valores

        # Realiza predição
        resultado = predizer(valores)

        # Limpa status e mostra título
        status_text.empty()

        # Mapeamento de resultados para exibir enquanto gera
        resultado_map = {
            1: ("Normal", "🟢", "Os parâmetros fetais estão dentro da normalidade"),
            2: ("Suspeito", "🟡", "Alguns parâmetros fetais requerem atenção"),
            3: ("Patológico", "🔴", "Parâmetros fetais indicam necessidade de intervenção")
        }
        status, emoji, descricao = resultado_map.get(resultado, ("Desconhecido", "⚪", "Resultado não identificado"))

        # Cabeçalho do resultado
        st.markdown(f"## {emoji} Diagnóstico: **{status}**")
        st.caption(descricao)
        st.markdown("---")

        # Container para streaming em tempo real - AGORA VISÍVEL!
        st.markdown("### 🤖 Análise Médica Detalhada")
        stream_container = st.empty()

        # Executa interpretação com streaming REAL
        explicacao_texto = interpretar(features_dict, resultado, stream_container)

        # Salva log
        salvar_log(features_dict, resultado, explicacao_texto)

        # Formata a resposta para histórico
        dados_formatados = formatar_resposta(resultado, explicacao_texto)

        # Exibe mensagem de sucesso
        st.success("✅ Análise concluída com sucesso!")

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
        st.write(explicacao.get('diagnostico', 'Não disponível'))
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
                st.write(f"**Explicação:** {exp_hist.get('diagnostico', 'Não disponível')}")
                st.write(f"**Urgência:** {exp_hist.get('urgencia', 'Não disponível')}")
                st.markdown("---")
else:
    st.info("👆 Preencha os dados acima e clique em 'Realizar Predição' para começar a análise.")