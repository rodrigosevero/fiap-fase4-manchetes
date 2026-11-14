# ESTADO ATUAL — fiap-fase4-manchetes

Atualizado em: 2026-11-14
Objetivo da entrega: prova substitutiva da Fase 4 (MLET) — gerar 3 manchetes em
português a partir de uma notícia, com modelo ajustado, playground Streamlit,
repositório no GitHub e vídeo de 5+ minutos.

## Decisões tomadas

| Tema | Decisão | Motivo |
| --- | --- | --- |
| Modelo | `unicamp-dl/ptt5-base-portuguese-vocab` | T5 220M só em português, cabe na T4 e roda em CPU |
| Prefixo | `manchete: ` no treino e na inferência | marcador consistente de tarefa |
| Entrada | até 3 parágrafos / 220 palavras | escolha inicial do experimento |
| Divisão | por grupo de acontecimento (TF-IDF ≥ 0,25 + título idêntico), 80/10/10 | evita vazamento de quase-duplicatas |
| Seleção de checkpoint | ROUGE-L na validação | validação decide; teste só no fim |
| Decodificação | 3 sugestões, 8 feixes, dedupe + amostragem para completar | feixe puro devolve frases quase iguais |
| Deploy | Hugging Face Spaces (Streamlit SDK) | 16 GB de RAM em CPU; o plano gratuito do Streamlit Cloud é apertado para 220M |
| Avaliação | ROUGE/BLEU + cópia literal + diversidade + memorização | ROUGE sozinho premia quem copia o texto |

## Comandos executados e resultados

```bash
# 1. Coleta (3000 notícias aceitas)
python3 scripts/coletar_noticias.py --quantidade 3000 --saida data/noticias.csv --pausa 2.5
# 3000 aceitas | 140 duplicadas | 285 descartadas | 122 requisições

# 2. Divisão por acontecimento
python3 scripts/preparar_dados.py
# 2827 grupos | 2400 treino / 300 validação / 300 teste | verificações de vazamento ok
```

Validações locais feitas:
- `scripts/treinar_modelo.py` rodou de ponta a ponta em CPU com 24 exemplos
  (`/tmp/modelo_teste`), gravou pesos, tokenizador e `metricas_treino.json`.
- `scripts/avaliar_modelo.py` rodou no modelo base e gravou JSON + previsões.
- `app.py` subiu com `streamlit run`, respondeu HTTP 200 e gerou sugestões pela UI.
- Corrigido bug em que `str.capitalize()` minusculizava nomes próprios.
- Corrigida deduplicação das 3 sugestões (feixe devolvia quase-duplicatas).

## Ambiente local

- Python 3.14.4, venv em `.venv` (criado com `virtualenv`, pois falta
  `python3-venv` no sistema).
- torch 2.15.0+cpu, transformers 5.23.0, streamlit 1.67.0.
- Sem GPU; inferência do T5-base leva ~3,7 s por notícia em CPU.

## Números reais obtidos

**Modelo base** (teste, 300 notícias): ROUGE-1 0,2954 | ROUGE-2 0,1413 |
ROUGE-L 0,2430 | BLEU 0,0460 | `taxa_copia_exata` 0,3481 |
`maior_trecho_comum` 0,8456 | 33,21 palavras por manchete.
Comportamento: não gera manchetes, copia o texto de entrada.

**Modelo ajustado** (teste, 300 notícias): ROUGE-1 0,4574 | ROUGE-2 0,2764 |
ROUGE-L 0,4209 | BLEU 0,1634 | `taxa_copia_exata` 0,0178 |
`maior_trecho_comum` 0,4812 | `distinct1` 0,4171 | 12,44 palavras por manchete |
`taxa_igual_a_titulo_de_treino` 0,0.

**Treino**: 2400 exemplos, 750 passos, 6 épocas, loss final 4,2453, 15,29 min.
**Validação** (seleção de checkpoint): ROUGE-1 0,4303 | ROUGE-L 0,3906 |
BLEU 0,1502 | loss 2,0835.
**Ambiente Colab**: Python 3.13.16, torch 2.14.1+cu130, transformers 5.22.1, T4.

**Modelo publicado**: <https://huggingface.co/rsevero/manchetes-ptt5>
(contém pesos, tokenizer, `metricas_treino.json` e `comparacao.json`).
**Repositório**: <https://github.com/rodrigosevero/fiap-fase4-manchetes>

Falhas documentadas: invenção factual ("Morre o papa João Paulo II, presidente de
Cuba"), 2 de 300 manchetes longas (até 41 palavras) e 16 de 900 sugestões copiadas.

## Próxima ação (em ordem)

### Situação

- Repositório publicado: <https://github.com/rodrigosevero/fiap-fase4-manchetes>
- Modelo no Hub: <https://huggingface.co/rsevero/manchetes-ptt5>
- App publicado e testado: <https://fiap-fase4-manchetes-xjjk25vdjk25m5pdcuwzjl.streamlit.app/>
  (Streamlit Community Cloud, secret `MODELO_ID = rsevero/manchetes-ptt5`)

### 1. Gravar o vídeo (pendente)

Seguir `ROTEIRO_VIDEO.md`, que já traz os números reais para ler na tela.
Publicar como "não listado" no YouTube e copiar o link.

### 2. Preencher o vídeo no `entrega.txt`

Trocar a linha `Vídeo (mínimo de 5 minutos): PENDENTE` pela URL.

### 3. Commit final (somente após confirmação do aluno)

Inclui: README com URL publicada, `entrega.txt` preenchido e `ESTADO_ATUAL.md`.

### 4. Opcional

- Apagar o Space do Hugging Face (`rsevero/manchetes-ptt5-playground`), que ficou
  parado por falta de cota na conta. O `Dockerfile` foi removido do repositório.
- Se sobrar tempo: ampliar a coleta para ~8000 notícias e retreinar, para reduzir
  a similaridade entre as três sugestões.

## Pendências e riscos

- Ainda não existe repositório no GitHub nem Space publicado.
- Token do Hugging Face necessário para publicar o modelo.
- Se a T4 estourar memória, reduzir `--batch` para 4 (mantendo `grad-accum`).
- Se o ROUGE do ajustado ficar abaixo do base, o culpado provável é a cópia
  literal: comparar `taxa_copia_exata` antes de concluir que piorou.
- Se o ajustado ainda copiar muito (`taxa_copia_exata` acima de ~0,15),
  o próximo passo é ampliar os dados: `--quantidade 8000` leva ~15 min de
  coleta, gerar novos splits e retreinar. O `ptt5` reage bem a mais dados.
