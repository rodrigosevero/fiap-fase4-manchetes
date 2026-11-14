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

### A. Publicar o repositório no GitHub (5 min, no VS Code)

```bash
git config user.name "SEU NOME"
git config user.email "seu@email.com"   # o mesmo e-mail da conta GitHub
git commit -m "Fase 4: coleta, splits por acontecimento, treino, avaliacao e app"
```

Depois crie o repositório **fiap-fase4-manchetes** (público, sem README e sem
.gitignore) em github.com/new e envie:

```bash
git remote add origin https://github.com/SEU_USUARIO/fiap-fase4-manchetes.git
git push -u origin main
```

No VS Code também dá para usar **Source Control → Publish Branch**, que resolve a
autenticação sozinho.

### B. Treinar no Colab (25 min)

1. Abra `notebooks/treinar_colab.ipynb` no Colab (aba GitHub) e escolha GPU T4.
2. Na célula 2, troque `USUARIO` pelo seu usuário do GitHub.
3. Execute as células 1 a 8 (a 9 e a 10 dependem do Hugging Face).
4. Baixe `resultados.zip` e copie `avaliacao/` e `metricas_treino.json` para o
   projeto local.

### C. Conta e token no Hugging Face (5 min)

1. Criar conta em <https://huggingface.co/join>.
2. <https://huggingface.co/settings/tokens> → **New token** → tipo **Write**.
3. No Colab, ícone de chave (Secrets) → nome `HF_TOKEN`, valor do token,
   habilitar acesso ao notebook.
4. Rodar as células 9 e 10 do notebook: publica o modelo e o `comparacao.json`.

### D. Publicar o Space (10 min)

1. <https://huggingface.co/new-space> → nome `manchetes-ptt5-playground`,
   SDK **Streamlit**, público.
2. Enviar os arquivos com `huggingface_hub` (não precisa de git):

```python
from huggingface_hub import HfApi
api = HfApi(token="hf_SEU_TOKEN")
api.upload_folder(
    repo_id="SEU_USUARIO/manchetes-ptt5-playground",
    repo_type="space",
    folder_path=".",
    allow_patterns=["app.py", "manchetes/*", "requirements.txt", "data/exemplos.json"],
)
```

3. No Space: **Settings → Variables and secrets → New secret**,
   nome `MODELO_ID`, valor `SEU_USUARIO/manchetes-ptt5`.
4. Esperar o build e conferir que gera 3 sugestões.

### E. Fechar a entrega

1. Preencher a seção de resultados do README com os números de
   `avaliacao/comparacao.json`.
2. Gravar o vídeo seguindo `ROTEIRO_VIDEO.md`.
3. Preencher `entrega.txt` com as três URLs.
4. `git add -A && git commit -m "resultados finais" && git push`.

## Pendências e riscos

- Ainda não existe repositório no GitHub nem Space publicado.
- Token do Hugging Face necessário para publicar o modelo.
- Se a T4 estourar memória, reduzir `--batch` para 4 (mantendo `grad-accum`).
- Se o ROUGE do ajustado ficar abaixo do base, o culpado provável é a cópia
  literal: comparar `taxa_copia_exata` antes de concluir que piorou.
- Se o ajustado ainda copiar muito (`taxa_copia_exata` acima de ~0,15),
  o próximo passo é ampliar os dados: `--quantidade 8000` leva ~15 min de
  coleta, gerar novos splits e retreinar. O `ptt5` reage bem a mais dados.
