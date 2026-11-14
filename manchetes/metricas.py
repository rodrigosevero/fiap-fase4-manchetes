"""Métricas de qualidade e de originalidade para as manchetes geradas."""

import csv

from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer

from manchetes.geracao import normalizar_texto

_ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)


def rouge(referencia, candidato):
    notas = _ROUGE.score(referencia, candidato)
    return {
        "rouge1": notas["rouge1"].fmeasure,
        "rouge2": notas["rouge2"].fmeasure,
        "rougeL": notas["rougeL"].fmeasure,
    }


def bleu(referencias, candidatos):
    if not candidatos:
        return 0.0
    nota = BLEU(effective_order=True).corpus_score(candidatos, [referencias])
    return nota.score / 100


def _tokens(texto):
    return normalizar_texto(texto).split()


def taxa_copia_exata(texto_fonte, manchete):
    """1.0 se a manchete aparece literalmente no texto-fonte (cópia direta)."""
    return 1.0 if normalizar_texto(manchete) in normalizar_texto(texto_fonte) else 0.0


def aproveitamento_fonte(texto_fonte, manchete):
    """Fração de palavras da manchete que já aparecem no texto-fonte."""
    palavras = _tokens(manchete)
    if not palavras:
        return 0.0
    palavras_fonte = set(_tokens(texto_fonte))
    return sum(1 for palavra in palavras if palavra in palavras_fonte) / len(palavras)


def maior_trecho_comum(texto_fonte, manchete):
    """Maior sequência contígua de palavras da manchete presente no texto-fonte."""
    alvo = _tokens(manchete)
    if not alvo:
        return 0.0
    fonte = " ".join(_tokens(texto_fonte))
    melhor = 0
    for inicio in range(len(alvo)):
        for fim in range(inicio + melhor + 1, len(alvo) + 1):
            if " ".join(alvo[inicio:fim]) not in fonte:
                break
            melhor = fim - inicio
    return melhor / len(alvo)


def distinct_n(sugestoes, n=1):
    """Diversidade entre as sugestões de uma mesma notícia (0 = todas iguais)."""
    gramas = []
    for sugestao in sugestoes:
        tokens = _tokens(sugestao)
        gramas.extend(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    if not gramas:
        return 0.0
    return len(set(gramas)) / len(gramas)


def jaccard(texto_a, texto_b):
    conjunto_a, conjunto_b = set(_tokens(texto_a)), set(_tokens(texto_b))
    if not conjunto_a or not conjunto_b:
        return 0.0
    return len(conjunto_a & conjunto_b) / len(conjunto_a | conjunto_b)


def avaliar_geracoes(registros, textos_fonte, titulos_de_treino=None):
    """Agrega métricas de um conjunto de gerações.

    registros: saída de gerar_para_registros ([{id, titulo_referencia, sugestoes}]).
    textos_fonte: mapa id -> entrada usada na geração.
    titulos_de_treino: opcional, títulos normalizados do treino para medir memorização.
    """
    if not registros:
        raise ValueError("nenhuma geração para avaliar")

    melhores, referencias = [], []
    copias, aproveitamentos, trechos, diversidades, jaccards, todas = [], [], [], [], [], []

    for registro in registros:
        fonte = textos_fonte[registro["id"]]
        sugestoes = registro["sugestoes"]
        if not sugestoes:
            continue
        todas.extend(sugestoes)
        referencias.append(registro["titulo_referencia"])
        notas = [rouge(registro["titulo_referencia"], sugestao) for sugestao in sugestoes]
        indice = max(range(len(notas)), key=lambda i: notas[i]["rougeL"])
        melhores.append((notas[indice], sugestoes[indice]))

        copias.extend(taxa_copia_exata(fonte, sugestao) for sugestao in sugestoes)
        aproveitamentos.extend(
            aproveitamento_fonte(fonte, sugestao) for sugestao in sugestoes
        )
        trechos.extend(maior_trecho_comum(fonte, sugestao) for sugestao in sugestoes)
        diversidades.append(distinct_n(sugestoes, 1))
        jaccards.extend(jaccard(registro["titulo_referencia"], s) for s in sugestoes)

    candidatos = [texto for _, texto in melhores]
    resultado = {
        "noticias_avaliadas": len(melhores),
        "sugestoes_por_noticia": round(len(todas) / len(melhores), 2),
        "rouge1": round(sum(n["rouge1"] for n, _ in melhores) / len(melhores), 4),
        "rouge2": round(sum(n["rouge2"] for n, _ in melhores) / len(melhores), 4),
        "rougeL": round(sum(n["rougeL"] for n, _ in melhores) / len(melhores), 4),
        "bleu": round(bleu(referencias, candidatos), 4),
        "jaccard_com_referencia": round(sum(jaccards) / len(jaccards), 4),
        "taxa_copia_exata": round(sum(copias) / len(copias), 4),
        "aproveitamento_fonte": round(sum(aproveitamentos) / len(aproveitamentos), 4),
        "maior_trecho_comum": round(sum(trechos) / len(trechos), 4),
        "distinct1_entre_sugestoes": round(sum(diversidades) / len(diversidades), 4),
        "palavras_por_manchete": round(
            sum(len(s.split()) for s in todas) / len(todas), 2
        ),
    }

    if titulos_de_treino is not None:
        repetidos = sum(
            1 for sugestao in todas if normalizar_texto(sugestao) in titulos_de_treino
        )
        resultado["taxa_igual_a_titulo_de_treino"] = round(repetidos / len(todas), 4)

    return resultado


def carregar_titulos_de_treino(caminho):
    with open(caminho, encoding="utf-8", newline="") as arquivo:
        return {normalizar_texto(linha["titulo"]) for linha in csv.DictReader(arquivo)}
