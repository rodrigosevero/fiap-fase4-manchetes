#!/usr/bin/env python3
"""Coleta notícias do Wikinews em português e prepara as entradas do modelo.

Uso:
    python scripts/coletar_noticias.py --quantidade 20 --saida data/noticias.csv

O CSV é salvo a cada lote e a coleta retoma a partir do arquivo existente,
sem repetir notícias já registradas.
"""

import argparse
import csv
import os
import random
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import requests

API_URL = "https://pt.wikinews.org/w/api.php"
USER_AGENT = (
    "fiap-fase4-manchetes/0.1 "
    "(projeto academico da FIAP; coleta de dados do Wikinews)"
)

QUANTIDADE_PADRAO = 1000
SAIDA_PADRAO = "data/noticias.csv"
TAMANHO_LOTE = 20  # limite do generator=random na API do MediaWiki
REQUISICOES_MAXIMAS_PADRAO = 400
TIMEOUT_PADRAO = 30.0
PAUSA_PADRAO = 2.0
TENTATIVAS_POR_REQUISICAO = 4
FALHAS_CONSECUTIVAS_MAXIMAS = 3
ESPERA_MAXIMA = 60.0

MIN_PALAVRAS = 40
MAX_PALAVRAS = 220
MAX_PARAGRAFOS = 3

COLUNAS = ["id", "titulo", "texto_original", "entrada", "url", "coletado_em"]

MESES = (
    "janeiro|fevereiro|março|abril|maio|junho|julho|agosto|"
    "setembro|outubro|novembro|dezembro"
)
DATA_SIMPLES = re.compile(
    rf"^(?:[A-Za-zÀ-ÿ]+,\s*)?\d{{1,2}}\s+de\s+(?:{MESES})\s+de\s+\d{{4}}$",
    re.IGNORECASE,
)
DATA_COM_LOCAL = re.compile(
    rf"^[^•]{{1,80}}•\s*\d{{1,2}}\s+de\s+(?:{MESES})\s+de\s+\d{{4}}$",
    re.IGNORECASE,
)
DATA_NUMERICA = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")

PONTUACAO_FRASE = re.compile(r"[.,;:!?]")
CREDITO = re.compile(
    r"^(?:(?:da|de|pela)\s+)?"
    r"(?:agência|agencia|redação|redacao|wikinews|wikimedia)\b",
    re.IGNORECASE,
)
CREDITO_POR = re.compile(r"^por\s+(.+)$", re.IGNORECASE)
PARTICULAS_NOME = {"de", "da", "do", "das", "dos", "e"}

STATUS_TRANSITORIOS = {408, 429, 500, 502, 503, 504}


def limpar_espacos(texto):
    texto = texto.replace("\u200b", "").replace("\xa0", " ")
    return re.sub(r"\s+", " ", texto).strip()


def normalizar_para_comparacao(texto):
    """Minúsculas, sem acentos e sem pontuação, para comparar textos."""
    texto = unicodedata.normalize("NFKD", limpar_espacos(texto).lower())
    texto = "".join(
        caractere for caractere in texto if not unicodedata.combining(caractere)
    )
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def eh_cabecalho_data(linha):
    return bool(
        DATA_SIMPLES.match(linha)
        or DATA_COM_LOCAL.match(linha)
        or DATA_NUMERICA.match(linha)
    )


def parece_nome_proprio(texto):
    palavras = texto.split()
    if not palavras:
        return False
    for palavra in palavras:
        if palavra.lower() in PARTICULAS_NOME:
            continue
        if not palavra[0].isupper():
            return False
    return True


def eh_credito_curto(linha):
    """Crédito de autoria ou agência: linha curta e sem pontuação de frase.

    A checagem de "Por" exige um nome próprio, para não remover frases
    jornalísticas que apenas começam com "Por".
    """
    if len(linha.split()) > 8 or PONTUACAO_FRASE.search(linha):
        return False
    if CREDITO.match(linha):
        return True
    autor = CREDITO_POR.match(linha)
    return bool(autor and parece_nome_proprio(autor.group(1)))


def preparar_entrada(texto_original):
    """Remove cabeçalho de data e créditos iniciais e limita parágrafos e palavras."""
    linhas = [limpar_espacos(linha) for linha in texto_original.splitlines()]
    linhas = [linha for linha in linhas if linha]

    paragrafos = []
    no_cabecalho = True
    for linha in linhas:
        if no_cabecalho and (eh_cabecalho_data(linha) or eh_credito_curto(linha)):
            continue
        no_cabecalho = False
        paragrafos.append(linha)

    entrada = " ".join(paragrafos[:MAX_PARAGRAFOS])
    palavras = entrada.split()
    if len(palavras) > MAX_PALAVRAS:
        entrada = " ".join(palavras[:MAX_PALAVRAS]).rstrip(" ,;:-")
    return limpar_espacos(entrada)


def contar_palavras(texto):
    return len(texto.split())


def montar_registro(pagina, pageids, textos):
    """Devolve (registro, chave_texto, motivo); registro é None se ignorado."""
    pageid = pagina.get("pageid")
    if not pageid:
        return None, None, "descartada"

    pageid = str(pageid)
    if pageid in pageids:
        return None, None, "duplicada"

    titulo = limpar_espacos(pagina.get("title") or "")
    if not titulo:
        return None, None, "descartada"

    texto_original = pagina.get("extract") or ""
    chave_texto = normalizar_para_comparacao(texto_original)
    if not chave_texto:
        return None, None, "descartada"
    if chave_texto in textos:
        return None, None, "duplicada"

    entrada = preparar_entrada(texto_original)
    if contar_palavras(entrada) < MIN_PALAVRAS:
        return None, None, "descartada"

    chave_titulo = normalizar_para_comparacao(titulo)
    if not chave_titulo or chave_titulo in normalizar_para_comparacao(entrada):
        return None, None, "descartada"

    registro = {
        "id": pageid,
        "titulo": titulo,
        "texto_original": texto_original,
        "entrada": entrada,
        "url": f"https://pt.wikinews.org/?curid={pageid}",
        "coletado_em": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    return registro, chave_texto, "aceita"


def carregar_registros(caminho):
    if not caminho.exists():
        return []

    with caminho.open("r", encoding="utf-8", newline="") as arquivo:
        leitor = csv.DictReader(arquivo)
        colunas = leitor.fieldnames or []
        faltantes = [coluna for coluna in COLUNAS if coluna not in colunas]
        if faltantes:
            raise SystemExit(
                f"CSV existente em {caminho} sem as colunas: {', '.join(faltantes)}"
            )
        return [
            {coluna: linha.get(coluna, "") for coluna in COLUNAS}
            for linha in leitor
            if linha.get("id")
        ]


def salvar_registros(caminho, registros):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_name(caminho.name + ".tmp")
    with temporario.open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS)
        escritor.writeheader()
        escritor.writerows(registros)
    os.replace(temporario, caminho)


def espera_da_resposta(resposta, pausa, tentativa):
    """Respeita o cabeçalho Retry-After e, sem ele, usa recuo exponencial."""
    cabecalho = resposta.headers.get("Retry-After", "").strip()
    if cabecalho:
        try:
            espera = float(cabecalho)
        except ValueError:
            espera = pausa * (2 ** (tentativa - 1))
    else:
        espera = pausa * (2 ** (tentativa - 1))
    return min(espera, ESPERA_MAXIMA) + random.uniform(0, 0.5)


def requisitar_lote(sessao, timeout, pausa):
    """Busca um lote de páginas aleatórias, com tentativas para falhas transitórias."""
    parametros = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "generator": "random",
        "grnnamespace": "0",
        "grnlimit": str(TAMANHO_LOTE),
        "prop": "extracts",
        "explaintext": "1",
        "exintro": "1",
    }

    for tentativa in range(1, TENTATIVAS_POR_REQUISICAO + 1):
        motivo = None
        espera = pausa * (2 ** (tentativa - 1))
        try:
            resposta = sessao.get(API_URL, params=parametros, timeout=timeout)
        except requests.RequestException as erro:
            motivo = f"{type(erro).__name__}: {erro}"
        else:
            if resposta.status_code == 200:
                try:
                    dados = resposta.json()
                except ValueError:
                    motivo = "resposta não é JSON válido"
                else:
                    if "error" in dados:
                        raise RuntimeError(f"a API retornou erro: {dados['error']}")
                    return dados.get("query", {}).get("pages") or []
            elif resposta.status_code in STATUS_TRANSITORIOS:
                motivo = f"HTTP {resposta.status_code}"
                espera = espera_da_resposta(resposta, pausa, tentativa)
            else:
                raise RuntimeError(
                    f"HTTP {resposta.status_code} não transitório: "
                    f"{resposta.text[:300]}"
                )

        if tentativa < TENTATIVAS_POR_REQUISICAO:
            print(f"  Falha ({motivo}); nova tentativa em {espera:.1f}s.")
            time.sleep(espera)
        else:
            print(f"  Falha após {TENTATIVAS_POR_REQUISICAO} tentativas: {motivo}")

    return None


def criar_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Coleta notícias do Wikinews em português e prepara entradas "
            "para o modelo."
        )
    )
    parser.add_argument(
        "--quantidade",
        type=int,
        default=QUANTIDADE_PADRAO,
        help=f"notícias aceitas desejadas (padrão: {QUANTIDADE_PADRAO})",
    )
    parser.add_argument(
        "--saida",
        default=SAIDA_PADRAO,
        help=f"caminho do CSV de saída (padrão: {SAIDA_PADRAO})",
    )
    parser.add_argument(
        "--max-requisicoes",
        type=int,
        default=REQUISICOES_MAXIMAS_PADRAO,
        help=(
            "limite total de requisições à API "
            f"(padrão: {REQUISICOES_MAXIMAS_PADRAO})"
        ),
    )
    parser.add_argument(
        "--pausa",
        type=float,
        default=PAUSA_PADRAO,
        help=f"pausa entre requisições em segundos (padrão: {PAUSA_PADRAO})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=TIMEOUT_PADRAO,
        help=f"timeout de cada requisição em segundos (padrão: {TIMEOUT_PADRAO})",
    )
    return parser


def main(argv=None):
    args = criar_parser().parse_args(argv)
    if args.quantidade < 1:
        raise SystemExit("--quantidade deve ser maior que zero")
    if args.max_requisicoes < 1:
        raise SystemExit("--max-requisicoes deve ser maior que zero")

    saida = Path(args.saida)
    registros = carregar_registros(saida)
    pageids = {registro["id"] for registro in registros}
    textos = {
        normalizar_para_comparacao(registro["texto_original"]) for registro in registros
    }

    print(f"Coletando até {args.quantidade} notícias aceitas de {API_URL}")
    print(f"Arquivo de saída: {saida}")
    if registros:
        print(f"Retomando a partir de {len(registros)} notícias já presentes no CSV.")
    if len(registros) >= args.quantidade:
        print("Meta já atingida; nada a coletar.")
        return 0

    sessao = requests.Session()
    sessao.headers.update({"User-Agent": USER_AGENT})

    aceitas = 0
    duplicadas = 0
    descartadas = 0
    requisicoes = 0
    falhas_seguidas = 0

    try:
        while len(registros) < args.quantidade and requisicoes < args.max_requisicoes:
            lote = requisitar_lote(sessao, args.timeout, args.pausa)
            requisicoes += 1

            if lote is None:
                falhas_seguidas += 1
                if falhas_seguidas >= FALHAS_CONSECUTIVAS_MAXIMAS:
                    print("Muitas falhas seguidas; interrompendo a coleta.")
                    break
            else:
                falhas_seguidas = 0
                novos = 0
                for pagina in lote:
                    if len(registros) >= args.quantidade:
                        break
                    registro, chave_texto, motivo = montar_registro(
                        pagina, pageids, textos
                    )
                    if registro is None:
                        if motivo == "duplicada":
                            duplicadas += 1
                        else:
                            descartadas += 1
                        continue
                    registros.append(registro)
                    pageids.add(registro["id"])
                    textos.add(chave_texto)
                    aceitas += 1
                    novos += 1

                if novos:
                    salvar_registros(saida, registros)

                print(
                    f"Lote {requisicoes}: {len(lote)} páginas recebidas | "
                    f"aceitas {len(registros)}/{args.quantidade} | "
                    f"duplicadas {duplicadas} | descartadas {descartadas}"
                )

            if len(registros) < args.quantidade:
                time.sleep(args.pausa)
    except KeyboardInterrupt:
        print("\nColeta interrompida; o progresso dos lotes já concluídos está salvo.")
        return 1

    print("\nResumo da coleta")
    print(f"  Meta: {args.quantidade} notícias aceitas")
    print(f"  Aceitas nesta execução: {aceitas}")
    print(f"  Duplicadas ignoradas: {duplicadas}")
    print(f"  Descartadas: {descartadas}")
    print(f"  Total no CSV: {len(registros)}")
    print(f"  Requisições realizadas: {requisicoes} (limite: {args.max_requisicoes})")

    if len(registros) < args.quantidade:
        if requisicoes >= args.max_requisicoes:
            print(
                "Limite de requisições atingido antes da meta: "
                f"{len(registros)} de {args.quantidade} notícias aceitas."
            )
        else:
            print(
                "Coleta interrompida antes da meta: "
                f"{len(registros)} de {args.quantidade} notícias aceitas."
            )
        return 1

    print(f"Meta atingida: {len(registros)} notícias aceitas em {saida}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
