#!/usr/bin/env python3
"""Avalia um modelo (base ou ajustado) no conjunto de teste.

Uso:
    python scripts/avaliar_modelo.py --modelo unicamp-dl/ptt5-base-portuguese-vocab \
        --rotulo base
    python scripts/avaliar_modelo.py --modelo modelos/manchetes-ptt5 --rotulo ajustado

Os dois comandos usam o mesmo conjunto, o mesmo prefixo e a mesma decodificação,
o que torna a comparação justa. Para comparar os JSONs gerados:

    python scripts/avaliar_modelo.py --comparar avaliacao/base.json avaliacao/ajustado.json
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from manchetes.geracao import carregar_modelo, carregar_tokenizador, gerar_para_registros
from manchetes.metricas import avaliar_geracoes, carregar_titulos_de_treino

TESTE_PADRAO = "data/splits/teste.csv"
TREINO_PADRAO = "data/splits/treino.csv"
PASTA_SAIDA_PADRAO = "avaliacao"


def carregar_teste(caminho, limite=None):
    with open(caminho, encoding="utf-8", newline="") as arquivo:
        registros = [
            {
                "id": linha["id"],
                "titulo": linha["titulo"],
                "entrada": linha["entrada"],
                "url": linha.get("url", ""),
            }
            for linha in csv.DictReader(arquivo)
        ]
    return registros[:limite] if limite else registros


def executar_avaliacao(args):
    registros = carregar_teste(args.teste, args.limite)
    print(f"Teste: {len(registros)} notícias | modelo: {args.modelo}")

    tokenizador = carregar_tokenizador(args.modelo)
    modelo = carregar_modelo(args.modelo)

    inicio = time.time()
    geracoes = gerar_para_registros(
        modelo,
        tokenizador,
        registros,
        quantidade=args.sequencias,
        num_beams=args.num_beams,
        amostragem=args.amostragem,
        temperatura=args.temperatura,
    )
    duracao = time.time() - inicio

    textos_fonte = {registro["id"]: registro["entrada"] for registro in registros}
    titulos_treino = (
        carregar_titulos_de_treino(args.treino) if Path(args.treino).exists() else None
    )
    notas = avaliar_geracoes(geracoes, textos_fonte, titulos_treino)
    notas.update(
        {
            "rotulo": args.rotulo,
            "modelo": args.modelo,
            "conjunto": str(args.teste),
            "decodificacao": {
                "num_beams": args.num_beams,
                "sequencias": args.sequencias,
                "amostragem": args.amostragem,
                "temperatura": args.temperatura,
                "com_prefixo": True,
            },
            "segundos_por_noticia": round(duracao / max(1, len(registros)), 3),
        }
    )

    pasta = Path(args.saida_dir)
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / f"{args.rotulo}.json").write_text(
        json.dumps(notas, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    caminho_previsoes = pasta / f"{args.rotulo}_previsoes.csv"
    with open(caminho_previsoes, "w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(
            ["id", "titulo_referencia", "sugestao_1", "sugestao_2", "sugestao_3"]
        )
        for geracao in geracoes:
            sugestoes = (geracao["sugestoes"] + ["", "", ""])[:3]
            escritor.writerow([geracao["id"], geracao["titulo_referencia"], *sugestoes])

    print(f"\nResultados ({args.rotulo})")
    for chave in (
        "rouge1", "rouge2", "rougeL", "bleu", "jaccard_com_referencia",
        "taxa_copia_exata", "aproveitamento_fonte", "maior_trecho_comum",
        "distinct1_entre_sugestoes", "palavras_por_manchete",
        "taxa_igual_a_titulo_de_treino", "segundos_por_noticia",
    ):
        if chave in notas:
            print(f"  {chave:30} {notas[chave]}")
    print(f"Arquivos em {pasta}/{args.rotulo}.json e {pasta}/{args.rotulo}_previsoes.csv")
    return notas


def comparar(caminhos):
    relatorios = [
        json.loads(Path(caminho).read_text(encoding="utf-8")) for caminho in caminhos
    ]
    chaves = [
        "rouge1", "rouge2", "rougeL", "bleu", "jaccard_com_referencia",
        "taxa_copia_exata", "aproveitamento_fonte", "maior_trecho_comum",
        "distinct1_entre_sugestoes", "palavras_por_manchete",
        "taxa_igual_a_titulo_de_treino", "segundos_por_noticia",
    ]
    rotulos = [relatorio["rotulo"] for relatorio in relatorios]

    largura = max(len(chave) for chave in chaves) + 2
    cabecalho = f"{'metrica':<{largura}}" + "".join(f"{rotulo:>14}" for rotulo in rotulos)
    print(cabecalho)
    print("-" * len(cabecalho))
    for chave in chaves:
        if not any(chave in relatorio for relatorio in relatorios):
            continue
        linha = f"{chave:<{largura}}"
        for relatorio in relatorios:
            valor = relatorio.get(chave)
            linha += f"{valor:>14}" if valor is not None else f"{'-':>14}"
        print(linha)

    tabela = {
        "rotulos": rotulos,
        "noticias_avaliadas": {
            relatorio["rotulo"]: relatorio["noticias_avaliadas"] for relatorio in relatorios
        },
        "metricas": {
            chave: {
                relatorio["rotulo"]: relatorio.get(chave) for relatorio in relatorios
            }
            for chave in chaves
            if any(chave in relatorio for relatorio in relatorios)
        },
    }
    destino = Path(PASTA_SAIDA_PADRAO) / "comparacao.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(
        json.dumps(tabela, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nComparação gravada em {destino}")
    return tabela


def main():
    parser = argparse.ArgumentParser(description="Avalia e compara modelos de manchetes.")
    parser.add_argument("--modelo", default="modelos/manchetes-ptt5")
    parser.add_argument("--rotulo", default="ajustado")
    parser.add_argument("--teste", default=TESTE_PADRAO)
    parser.add_argument("--treino", default=TREINO_PADRAO)
    parser.add_argument("--saida-dir", default=PASTA_SAIDA_PADRAO)
    parser.add_argument("--sequencias", type=int, default=3)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--amostragem", action="store_true")
    parser.add_argument("--temperatura", type=float, default=0.9)
    parser.add_argument("--limite", type=int, default=None)
    parser.add_argument(
        "--comparar",
        nargs="+",
        default=None,
        help="apenas compara JSONs de avaliação já gerados",
    )
    args = parser.parse_args()

    if args.comparar:
        comparar(args.comparar)
        return 0

    executar_avaliacao(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
