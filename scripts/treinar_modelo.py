#!/usr/bin/env python3
"""Ajusta um modelo T5 pré-treinado em português para gerar manchetes.

Pensado para rodar no Google Colab com GPU T4, mas também funciona em CPU para
testes rápidos:

    python scripts/treinar_modelo.py --epocas 1 --limite-exemplos 40

Lê data/splits/{treino,validacao}.csv e grava o modelo em modelos/manchetes-ptt5.
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from manchetes.geracao import (
    PREFIXO_TAREFA,
    carregar_modelo,
    carregar_tokenizador,
    gerar_manchetes,
)

MODELO_PADRAO = "unicamp-dl/ptt5-base-portuguese-vocab"
DADOS_PADRAO = "data/splits"
SAIDA_PADRAO = "modelos/manchetes-ptt5"


def carregar_split(caminho, limite=None):
    import csv

    with open(caminho, encoding="utf-8", newline="") as arquivo:
        registros = [
            {"id": linha["id"], "titulo": linha["titulo"], "entrada": linha["entrada"]}
            for linha in csv.DictReader(arquivo)
        ]
    return registros[:limite] if limite else registros


class DatasetManchetes(torch.utils.data.Dataset):
    def __init__(self, registros, tokenizador, max_entrada, max_saida):
        self.itens = []
        for registro in registros:
            entrada = tokenizador(
                PREFIXO_TAREFA + registro["entrada"],
                max_length=max_entrada,
                truncation=True,
            )
            alvo = tokenizador(
                registro["titulo"],
                max_length=max_saida,
                truncation=True,
            )
            self.itens.append(
                {
                    "input_ids": entrada["input_ids"],
                    "attention_mask": entrada["attention_mask"],
                    "labels": alvo["input_ids"],
                }
            )

    def __len__(self):
        return len(self.itens)

    def __getitem__(self, indice):
        return self.itens[indice]


def criar_argumentos(args, passos_totais):
    """Monta TrainingArguments aceitando nomes de parâmetro de transformers 4 e 5."""
    comuns = {
        "output_dir": str(Path(args.saida) / "checkpoints"),
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.batch,
        "per_device_eval_batch_size": args.batch,
        "gradient_accumulation_steps": args.grad_accum,
        "weight_decay": 0.01,
        "num_train_epochs": args.epocas,
        "warmup_steps": max(1, int(0.1 * passos_totais)),
        "lr_scheduler_type": "linear",
        "logging_steps": 50,
        "save_total_limit": 2,
        "predict_with_generate": True,
        "generation_max_length": args.max_saida,
        "generation_num_beams": 4,
        "load_best_model_at_end": True,
        "metric_for_best_model": "rougeL",
        "greater_is_better": True,
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "fp16": args.fp16,
        "report_to": [],
        "seed": args.semente,
        "data_seed": args.semente,
    }
    try:
        return Seq2SeqTrainingArguments(**comuns)
    except TypeError:
        comuns.pop("eval_strategy")
        comuns["evaluation_strategy"] = "epoch"
        return Seq2SeqTrainingArguments(**comuns)


def criar_trainer(args, nome_modelo):
    tokenizador = carregar_tokenizador(nome_modelo)
    modelo = carregar_modelo(nome_modelo)

    treino = carregar_split(Path(args.dados) / "treino.csv", args.limite_exemplos)
    validacao = carregar_split(Path(args.dados) / "validacao.csv", args.limite_validacao)
    print(f"Treino: {len(treino)} exemplos | Validação: {len(validacao)} exemplos")

    dataset_treino = DatasetManchetes(treino, tokenizador, args.max_entrada, args.max_saida)
    dataset_validacao = DatasetManchetes(
        validacao, tokenizador, args.max_entrada, args.max_saida
    )

    passos_por_epoca = max(
        1, len(dataset_treino) // (args.batch * args.grad_accum)
    )
    argumentos = criar_argumentos(args, passos_por_epoca * args.epocas)

    colador = DataCollatorForSeq2Seq(tokenizador, model=modelo, padding=True)

    def metricas(previsoes):
        from rouge_score import rouge_scorer
        from sacrebleu.metrics import BLEU

        ids_previstos = np.where(
            previsoes.predictions != -100, previsoes.predictions, tokenizador.pad_token_id
        )
        textos_previstos = tokenizador.batch_decode(
            ids_previstos, skip_special_tokens=True
        )
        rotulos = np.where(
            previsoes.label_ids != -100, previsoes.label_ids, tokenizador.pad_token_id
        )
        textos_rotulos = tokenizador.batch_decode(rotulos, skip_special_tokens=True)

        avaliador = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=False
        )
        notas = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        for previsto, rotulo in zip(textos_previstos, textos_rotulos):
            parcial = avaliador.score(rotulo, previsto)
            for chave in notas:
                notas[chave] += parcial[chave].fmeasure
        notas = {chave: valor / len(textos_previstos) for chave, valor in notas.items()}
        notas["bleu"] = BLEU(effective_order=True).corpus_score(
            textos_previstos, [textos_rotulos]
        ).score / 100
        return notas

    opcionais = {
        "model": modelo,
        "args": argumentos,
        "train_dataset": dataset_treino,
        "eval_dataset": dataset_validacao,
        "data_collator": colador,
        "compute_metrics": metricas,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=2)],
    }
    try:
        trainer = Seq2SeqTrainer(processing_class=tokenizador, **opcionais)
    except TypeError:
        trainer = Seq2SeqTrainer(tokenizer=tokenizador, **opcionais)

    return trainer, tokenizador, modelo


def main():
    parser = argparse.ArgumentParser(description="Ajusta o T5 para gerar manchetes.")
    parser.add_argument("--modelo", default=MODELO_PADRAO)
    parser.add_argument("--dados", default=DADOS_PADRAO)
    parser.add_argument("--saida", default=SAIDA_PADRAO)
    parser.add_argument("--epocas", type=float, default=6)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-entrada", type=int, default=512)
    parser.add_argument("--max-saida", type=int, default=64)
    parser.add_argument("--semente", type=int, default=42)
    parser.add_argument("--limite-exemplos", type=int, default=None)
    parser.add_argument("--limite-validacao", type=int, default=100)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--push-hub", default=None, help="repositório no Hugging Face")
    args = parser.parse_args()

    inicio = time.time()
    trainer, tokenizador, _ = criar_trainer(args, args.modelo)
    resultado = trainer.train()

    pasta = Path(args.saida)
    pasta.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(pasta))
    tokenizador.save_pretrained(str(pasta))

    notas_validacao = trainer.evaluate()
    notas = {
        chave.replace("eval_", ""): valor
        for chave, valor in notas_validacao.items()
        if chave.startswith("eval_") and isinstance(valor, (int, float))
    }

    amostras = carregar_split(Path(args.dados) / "validacao.csv", 3)
    exemplos = []
    for registro in amostras:
        exemplos.append(
            {
                "entrada": registro["entrada"][:300],
                "titulo_referencia": registro["titulo"],
                "sugestoes": gerar_manchetes(
                    trainer.model, tokenizador, registro["entrada"], quantidade=3
                ),
            }
        )

    historico = [
        {chave: valor for chave, valor in linha.items()}
        for linha in trainer.state.log_history
    ]
    relatorio = {
        "modelo_base": args.modelo,
        "hiperparametros": {
            "epocas": args.epocas,
            "batch": args.batch,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "max_entrada": args.max_entrada,
            "max_saida": args.max_saida,
            "semente": args.semente,
            "fp16": args.fp16,
        },
        "treino": {
            "exemplos": len(trainer.train_dataset),
            "passos": resultado.global_step,
            "loss_final": round(resultado.training_loss, 4),
            "minutos": round((time.time() - inicio) / 60, 2),
        },
        "validacao": notas,
        "melhor_checkpoint": trainer.state.best_model_checkpoint,
        "historico": historico,
        "exemplos_gerados": exemplos,
        "ambiente": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "transformers": __import__("transformers").__version__,
        },
    }
    (pasta / "metricas_treino.json").write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("\nMétricas de validação:")
    for chave, valor in notas.items():
        if isinstance(valor, float):
            print(f"  {chave:8} {valor:.4f}")
        else:
            print(f"  {chave:8} {valor}")
    print(f"Modelo salvo em {pasta}")

    if args.push_hub:
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("HF_TOKEN não definido; envio para o Hub ignorado.")
        else:
            from huggingface_hub import HfApi

            trainer.model.push_to_hub(args.push_hub, token=token)
            tokenizador.push_to_hub(args.push_hub, token=token)
            HfApi().upload_file(
                path_or_fileobj=str(pasta / "metricas_treino.json"),
                path_in_repo="metricas_treino.json",
                repo_id=args.push_hub,
                token=token,
            )
            print(f"Modelo publicado em https://huggingface.co/{args.push_hub}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
