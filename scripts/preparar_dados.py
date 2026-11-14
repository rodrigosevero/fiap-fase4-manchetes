#!/usr/bin/env python3
"""Agrupa notícias do mesmo acontecimento e divide em treino/validação/teste.

Uso:
    python scripts/preparar_dados.py

Lê data/noticias.csv e grava data/splits/{treino,validacao,teste}.csv mais
data/splits/resumo.json. A divisão é feita por grupo de acontecimento, nunca por
notícia isolada, para que textos sobre o mesmo evento não atravessem os conjuntos.
"""

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ENTRADA_PADRAO = "data/noticias.csv"
PASTA_SPLITS_PADRAO = "data/splits"
SEMENTE_PADRAO = 42
LIMIAR_SIMILARIDADE_PADRAO = 0.25
PROPORCOES = {"treino": 0.80, "validacao": 0.10, "teste": 0.10}

COLUNAS_SPLIT = ["id", "titulo", "entrada", "texto_original", "url", "grupo", "data"]

MESES = {
    "janeiro": 1, "fevereiro": 2, "março": 3, "abril": 4, "maio": 5, "junho": 6,
    "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11,
    "dezembro": 12,
}
DATA_TEXTO = re.compile(r"(\d{1,2})\s+de\s+([A-Za-zÀ-ÿ]+)\s+de\s+(\d{4})")


def normalizar(texto):
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def extrair_data(texto_original):
    """Lê a data de publicação do cabeçalho da introdução, quando presente."""
    for linha in texto_original.splitlines()[:4]:
        encontrado = DATA_TEXTO.search(linha)
        if not encontrado:
            continue
        dia, mes, ano = encontrado.groups()
        mes = MESES.get(mes.lower())
        if mes:
            try:
                return date(int(ano), mes, int(dia)).isoformat()
            except ValueError:
                return ""
    return ""


def carregar_noticias(caminho):
    with caminho.open("r", encoding="utf-8", newline="") as arquivo:
        linhas = [
            linha for linha in csv.DictReader(arquivo)
            if (linha.get("entrada") or "").strip()
            and (linha.get("titulo") or "").strip()
        ]
    return linhas


class UniaoBusca:
    def __init__(self, tamanho):
        self.pai = list(range(tamanho))

    def encontrar(self, item):
        while self.pai[item] != item:
            self.pai[item] = self.pai[self.pai[item]]
            item = self.pai[item]
        return item

    def unir(self, primeiro, segundo):
        raiz_a, raiz_b = self.encontrar(primeiro), self.encontrar(segundo)
        if raiz_a != raiz_b:
            self.pai[raiz_b] = raiz_a


def agrupar_acontecimentos(noticias, limiar):
    """Agrupa por título normalizado idêntico ou por similaridade TF-IDF alta."""
    uniao = UniaoBusca(len(noticias))

    primeiros_por_titulo = {}
    for indice, noticia in enumerate(noticias):
        chave = normalizar(noticia["titulo"])
        if chave in primeiros_por_titulo:
            uniao.unir(primeiros_por_titulo[chave], indice)
        else:
            primeiros_por_titulo[chave] = indice

    documentos = [
        f"{noticia['titulo']} {noticia['entrada']}" for noticia in noticias
    ]
    vetorizador = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.6,
        strip_accents="unicode",
    )
    matriz = vetorizador.fit_transform(documentos)
    similaridade = cosine_similarity(matriz)
    np.fill_diagonal(similaridade, 0.0)

    linhas, colunas = np.where(similaridade >= limiar)
    for primeiro, segundo in zip(linhas, colunas):
        if primeiro < segundo:
            uniao.unir(int(primeiro), int(segundo))

    raizes = [uniao.encontrar(indice) for indice in range(len(noticias))]
    rotulos = {}
    grupos = []
    for raiz in raizes:
        if raiz not in rotulos:
            rotulos[raiz] = f"g{len(rotulos):05d}"
        grupos.append(rotulos[raiz])
    return grupos, similaridade


def dividir_por_grupo(noticias, grupos, semente):
    """Distribui grupos inteiros entre os conjuntos, equilibrando as quantidades."""
    tamanho_por_grupo = Counter(grupos)
    rng = np.random.default_rng(semente)
    grupos_embaralhados = list(tamanho_por_grupo)
    rng.shuffle(grupos_embaralhados)
    # Grupos maiores primeiro para reduzir o desvio das proporções alvo.
    grupos_embaralhados.sort(key=lambda grupo: tamanho_por_grupo[grupo], reverse=True)

    total = len(noticias)
    alvo = {nome: proporcao * total for nome, proporcao in PROPORCOES.items()}
    atribuido = {nome: 0 for nome in PROPORCOES}
    grupo_para_split = {}

    for grupo in grupos_embaralhados:
        # Enquanto todos estão abaixo do alvo, o maior déficit indica o destino.
        deficits = {nome: alvo[nome] - atribuido[nome] for nome in PROPORCOES}
        destino = max(deficits, key=lambda nome: deficits[nome])
        grupo_para_split[grupo] = destino
        atribuido[destino] += tamanho_por_grupo[grupo]

    return grupo_para_split


def verificar_divisao(registros_por_split):
    """Confere que nenhum grupo, id ou texto aparece em mais de um conjunto."""
    grupos_por_split = {
        nome: {registro["grupo"] for registro in registros}
        for nome, registros in registros_por_split.items()
    }
    nomes = list(registros_por_split)
    for i, primeiro in enumerate(nomes):
        for segundo in nomes[i + 1:]:
            repetidos = grupos_por_split[primeiro] & grupos_por_split[segundo]
            if repetidos:
                raise SystemExit(
                    f"Vazamento: grupos em '{primeiro}' e '{segundo}': "
                    f"{sorted(repetidos)[:5]}"
                )

    ids = Counter()
    textos = Counter()
    for registros in registros_por_split.values():
        for registro in registros:
            ids[registro["id"]] += 1
            textos[normalizar(registro["entrada"])] += 1
    if qualquer := [chave for chave, total in ids.items() if total > 1]:
        raise SystemExit(f"ids repetidos entre conjuntos: {qualquer[:5]}")
    if qualquer := [chave for chave, total in textos.items() if total > 1]:
        raise SystemExit(f"textos repetidos entre conjuntos: {qualquer[:3]}")


def salvar_split(caminho, registros):
    with caminho.open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS_SPLIT)
        escritor.writeheader()
        for registro in registros:
            escritor.writerow(
                {coluna: registro.get(coluna, "") for coluna in COLUNAS_SPLIT}
            )


def main():
    parser = argparse.ArgumentParser(
        description="Agrupa por acontecimento e divide os dados."
    )
    parser.add_argument("--entrada", default=ENTRADA_PADRAO)
    parser.add_argument("--pasta-saida", default=PASTA_SPLITS_PADRAO)
    parser.add_argument("--semente", type=int, default=SEMENTE_PADRAO)
    parser.add_argument(
        "--limiar",
        type=float,
        default=LIMIAR_SIMILARIDADE_PADRAO,
        help=(
            "similaridade TF-IDF mínima para tratar duas notícias "
            "como mesmo acontecimento"
        ),
    )
    args = parser.parse_args()

    noticias = carregar_noticias(Path(args.entrada))
    if not noticias:
        raise SystemExit(f"nenhuma notícia utilizável em {args.entrada}")

    vistos, unicas = set(), []
    for noticia in noticias:
        chave = normalizar(noticia["entrada"])
        if chave in vistos:
            continue
        vistos.add(chave)
        unicas.append(noticia)
    print(f"Notícias lidas: {len(noticias)} | após remover textos repetidos: {len(unicas)}")

    grupos, similaridade = agrupar_acontecimentos(unicas, args.limiar)
    tamanhos = Counter(grupos)
    print(
        f"Grupos de acontecimento: {len(tamanhos)} | "
        f"maior grupo: {max(tamanhos.values())} | "
        f"grupos com 1 notícia: {sum(1 for t in tamanhos.values() if t == 1)}"
    )

    grupo_para_split = dividir_por_grupo(unicas, grupos, args.semente)
    registros_por_split = {nome: [] for nome in PROPORCOES}
    for noticia, grupo in zip(unicas, grupos):
        registros_por_split[grupo_para_split[grupo]].append(
            {
                "id": noticia["id"],
                "titulo": noticia["titulo"],
                "entrada": noticia["entrada"],
                "texto_original": noticia["texto_original"],
                "url": noticia["url"],
                "grupo": grupo,
                "data": extrair_data(noticia["texto_original"]),
            }
        )

    verificar_divisao(registros_por_split)

    pasta = Path(args.pasta_saida)
    pasta.mkdir(parents=True, exist_ok=True)
    for nome, registros in registros_por_split.items():
        registros.sort(key=lambda registro: (registro["data"], registro["id"]))
        salvar_split(pasta / f"{nome}.csv", registros)
        grupos_no_split = len({registro["grupo"] for registro in registros})
        print(
            f"  {nome:10} {len(registros):5} notícias | {grupos_no_split:5} grupos | "
            f"{len(registros) / len(unicas):.1%}"
        )

    resumo = {
        "entrada": str(args.entrada),
        "noticias_lidas": len(noticias),
        "noticias_unicas": len(unicas),
        "removidas_por_texto_repetido": len(noticias) - len(unicas),
        "grupos": len(tamanhos),
        "maior_grupo": max(tamanhos.values()),
        "limiar_similaridade": args.limiar,
        "semente": args.semente,
        "splits": {
            nome: {
                "noticias": len(registros),
                "grupos": len({registro["grupo"] for registro in registros}),
                "fracao": round(len(registros) / len(unicas), 4),
            }
            for nome, registros in registros_por_split.items()
        },
        "verificacoes": {
            "grupos_disjuntos": True,
            "ids_unicos_entre_splits": True,
            "textos_unicos_entre_splits": True,
        },
    }
    (pasta / "resumo.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Resumo gravado em {pasta / 'resumo.json'}")
    print("Verificações: grupos, ids e textos disjuntos entre treino, validação e teste.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
