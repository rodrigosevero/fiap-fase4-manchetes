"""Playground Streamlit: gera manchetes em português a partir de um trecho de notícia."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from manchetes.geracao import carregar_modelo, carregar_tokenizador, gerar_manchetes

MODELO_PADRAO = "modelos/manchetes-ptt5"
MODELO_BASE = "unicamp-dl/ptt5-base-portuguese-vocab"
EXEMPLOS_PADRAO = "data/exemplos.json"
CAMINHOS_RESULTADO = [
    "avaliacao/comparacao.json",
    "avaliacao/ajustado.json",
    "metricas_treino.json",
]

st.set_page_config(page_title="Manchetes PT — playground", page_icon="📰", layout="wide")


def identificar_modelo():
    """Usa a variável MODELO_ID/Space secret e cai para o caminho local do projeto."""
    escolhido = os.environ.get("MODELO_ID", "").strip()
    if escolhido:
        return escolhido
    return MODELO_PADRAO if Path(MODELO_PADRAO).exists() else MODELO_BASE


@st.cache_resource(show_spinner="Carregando o modelo...")
def obter_modelo(identificador):
    tokenizador = carregar_tokenizador(identificador)
    modelo = carregar_modelo(identificador)
    return modelo, tokenizador


def carregar_exemplos():
    caminho = Path(EXEMPLOS_PADRAO)
    if caminho.exists():
        return json.loads(caminho.read_text(encoding="utf-8"))
    return []


def carregar_resultados():
    for caminho in CAMINHOS_RESULTADO:
        if Path(caminho).exists():
            return json.loads(Path(caminho).read_text(encoding="utf-8")), caminho
    return None, None


def mostrar_metricas():
    dados, origem = carregar_resultados()
    if dados is None:
        st.caption("Métricas de avaliação ainda não disponíveis neste ambiente.")
        return

    st.caption(f"Fonte: `{origem}`")
    if "metricas" in dados and "rotulos" in dados:
        linhas = []
        for metrica, valores in dados["metricas"].items():
            if metrica in {"segundos_por_noticia"}:
                continue
            linhas.append({"métrica": metrica, **valores})
        st.dataframe(linhas, use_container_width=True, hide_index=True)
        st.caption(
            "ROUGE e BLEU medem proximidade com o título original. "
            "`taxa_copia_exata` e `maior_trecho_comum` medem quanto do texto-fonte foi "
            "reaproveitado literalmente."
        )
    else:
        colunas = st.columns(3)
        for coluna, chave in zip(colunas, ("rouge1", "rougeL", "bleu")):
            if chave in dados:
                coluna.metric(chave.upper(), f"{dados[chave]:.3f}")


def main():
    identificador = identificar_modelo()

    st.title("📰 Gerador de manchetes em português")
    st.write(
        "Cole o trecho de uma notícia e receba **três sugestões de manchete**. "
        "O modelo é um T5 pré-treinado em português e ajustado em notícias do "
        "Wikinotícias."
    )

    with st.sidebar:
        st.header("Modelo")
        st.code(identificador, language=None)
        st.caption(
            "Base: `unicamp-dl/ptt5-base-portuguese-vocab`, ajustada para gerar manchetes."
            if identificador != MODELO_BASE
            else "Modelo base sem ajuste: as sugestões tendem a ser genéricas."
        )
        st.header("Métricas no conjunto de teste")
        mostrar_metricas()
        st.header("Limitações")
        st.markdown(
            "- As manchetes podem repetir trechos do texto ou inventar detalhes.\n"
            "- O modelo não verifica fatos nem conhece eventos recentes.\n"
            "- Títulos de séries esportivas e de eleições saem em formato repetido.\n"
            "- Sempre revise antes de publicar."
        )

    exemplos = carregar_exemplos()
    if exemplos:
        rotulos = [exemplo["rotulo"] for exemplo in exemplos]
        escolhido = st.selectbox("Exemplos de notícia", ["(digitar texto)"] + rotulos)
        texto_inicial = ""
        if escolhido != "(digitar texto)":
            texto_inicial = next(
                exemplo["entrada"] for exemplo in exemplos if exemplo["rotulo"] == escolhido
            )
    else:
        texto_inicial = ""

    with st.form("entrada"):
        texto = st.text_area(
            "Trecho da notícia",
            value=texto_inicial,
            height=260,
            placeholder="Cole aqui o texto da notícia...",
        )
        colunas = st.columns(3)
        quantidade = colunas[0].slider("Sugestões", 1, 3, 3)
        num_beams = colunas[1].slider("Feixes de busca", 1, 8, 4)
        comprimento = colunas[2].slider("Máximo de tokens", 12, 64, 48)
        amostragem = st.checkbox(
            "Usar amostragem (mais criativo, menos fiel)", value=False
        )
        temperatura = st.slider("Temperatura", 0.5, 1.5, 0.9, 0.1, disabled=not amostragem)
        enviado = st.form_submit_button("Gerar manchetes", type="primary")

    if enviado:
        if len(texto.split()) < 15:
            st.warning("Informe um trecho com pelo menos 15 palavras para haver contexto.")
            return

        with st.spinner("Gerando..."):
            modelo, tokenizador = obter_modelo(identificador)
            sugestoes = gerar_manchetes(
                modelo,
                tokenizador,
                texto,
                quantidade=quantidade,
                num_beams=num_beams,
                comprimento_maximo=comprimento,
                temperatura=temperatura,
                amostragem=amostragem,
            )

        st.subheader("Sugestões")
        for indice, sugestao in enumerate(sugestoes, start=1):
            st.markdown(f"**{indice}.** {sugestao}")

        st.caption(
            f"{len(texto.split())} palavras de entrada · "
            f"{'amostragem' if amostragem else 'busca em feixe'}"
        )

    with st.expander("Como o modelo foi preparado"):
        st.markdown(
            "1. Coleta das notícias pela API pública do Wikinotícias.\n"
            "2. Remoção de cabeçalhos de data e créditos de agência.\n"
            "3. Uso de até três parágrafos e 220 palavras como entrada.\n"
            "4. Divisão por acontecimento entre treino, validação e teste.\n"
            "5. Ajuste do T5 com Hugging Face Transformers e prefixo `manchete: `."
        )


if __name__ == "__main__":
    main()
