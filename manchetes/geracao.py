"""Geração de manchetes com um modelo T5 ajustado para português."""

import re
import unicodedata

PREFIXO_TAREFA = "manchete: "
PREFIXO_SAIDA = re.compile(r"^\s*man\s*chete\s*[:\-.,]\s*", re.IGNORECASE)


def normalizar_texto(texto):
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def preparar_entrada(texto, limite_caracteres=6000):
    """Deixa o trecho de notícia em uma linha só e limita o tamanho enviado."""
    linhas = [re.sub(r"\s+", " ", linha).strip() for linha in texto.splitlines()]
    texto = " ".join(linha for linha in linhas if linha)
    return texto[:limite_caracteres]


def montar_entrada(texto, com_prefixo=True):
    entrada = preparar_entrada(texto)
    return f"{PREFIXO_TAREFA}{entrada}" if com_prefixo else entrada


def limpar_manchete(texto):
    """Remove o prefixo ecoado e ajusta só a primeira letra, sem mexer no resto."""
    texto = PREFIXO_SAIDA.sub("", texto.strip())
    texto = re.sub(r"\s+", " ", texto).strip()
    if texto and texto[0].islower():
        texto = texto[0].upper() + texto[1:]
    return texto


def carregar_tokenizador(nome_ou_caminho):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(nome_ou_caminho)


def carregar_modelo(nome_ou_caminho):
    from transformers import AutoModelForSeq2SeqLM

    modelo = AutoModelForSeq2SeqLM.from_pretrained(nome_ou_caminho)
    # O generation_config do T5 traz max_length=20, o que gera avisos a cada chamada.
    modelo.generation_config.max_length = None
    modelo.generation_config.min_length = None
    modelo.eval()
    return modelo


def _decodificar(modelo, tokenizador, texto, parametros):
    import torch

    lote = tokenizador(
        montar_entrada(texto), return_tensors="pt", truncation=True, max_length=512
    )
    lote = {chave: valor.to(modelo.device) for chave, valor in lote.items()}
    with torch.no_grad():
        saida = modelo.generate(**lote, **parametros)
    return tokenizador.batch_decode(saida, skip_special_tokens=True)


def gerar_manchetes(
    modelo,
    tokenizador,
    texto,
    quantidade=3,
    num_beams=4,
    comprimento_maximo=48,
    temperatura=0.9,
    amostragem=False,
):
    """Gera sugestões de manchete distintas para um trecho de notícia.

    A busca em feixe costuma devolver frases quase idênticas. Por isso os
    candidatos são deduplicados e, se faltarem sugestões, completa-se com
    amostragem aleatória.
    """
    base = {
        "max_new_tokens": comprimento_maximo,
        "min_new_tokens": 4,
        "no_repeat_ngram_size": 3,
        "length_penalty": 1.0,
    }

    if amostragem:
        parametros = {
            **base,
            "do_sample": True,
            "top_p": 0.92,
            "temperature": temperatura,
            "num_return_sequences": quantidade * 2,
        }
    else:
        parametros = {
            **base,
            "num_beams": max(num_beams, quantidade * 2),
            "num_return_sequences": quantidade * 2,
        }

    sugestoes, vistos = [], set()
    for manchete in map(limpar_manchete, _decodificar(
        modelo, tokenizador, texto, parametros
    )):
        chave = normalizar_texto(manchete)
        if not chave or chave in vistos:
            continue
        vistos.add(chave)
        sugestoes.append(manchete)
        if len(sugestoes) == quantidade:
            return sugestoes

    completar = {
        **base,
        "do_sample": True,
        "top_p": 0.95,
        "temperature": max(temperatura, 1.0),
        "num_return_sequences": quantidade * 2,
    }
    for manchete in map(limpar_manchete, _decodificar(
        modelo, tokenizador, texto, completar
    )):
        chave = normalizar_texto(manchete)
        if not chave or chave in vistos:
            continue
        vistos.add(chave)
        sugestoes.append(manchete)
        if len(sugestoes) == quantidade:
            break

    return sugestoes


def gerar_para_registros(modelo, tokenizador, registros, quantidade=3, **kwargs):
    """Gera manchetes para vários registros e devolve uma lista de dicionários."""
    resultados = []
    for registro in registros:
        sugestoes = gerar_manchetes(
            modelo, tokenizador, registro["entrada"], quantidade=quantidade, **kwargs
        )
        resultados.append(
            {
                "id": registro["id"],
                "titulo_referencia": registro["titulo"],
                "sugestoes": sugestoes,
            }
        )
    return resultados
