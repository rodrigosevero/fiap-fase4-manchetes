# fiap-fase4-manchetes

Projeto da prova substitutiva da Fase 4 de Machine Learning Engineering da FIAP.

## Objetivo

Ajustar um modelo pré-treinado do Hugging Face Transformers para **gerar três
sugestões de manchete em português** a partir do texto de uma notícia, entregando
um playground em Streamlit, o repositório no GitHub e um vídeo de pelo menos cinco
minutos.

O desenvolvimento, a coleta, a avaliação e o Streamlit rodam no VS Code com 16 GB
de RAM e sem GPU NVIDIA. O treinamento roda no Google Colab com GPU Tesla T4,
reaproveitando o código deste repositório.

> **Status:** coleta, preparação, divisão por acontecimento, script de treino,
> script de avaliação e playground estão implementados e validados localmente.
> O ajuste do modelo depende da execução no Colab — veja
> [notebooks/treinar_colab.ipynb](notebooks/treinar_colab.ipynb).
> O acompanhamento da entrega está em [ESTADO_ATUAL.md](ESTADO_ATUAL.md).

## Estrutura

```
fiap-fase4-manchetes/
├── app.py                        # playground Streamlit
├── manchetes/
│   ├── geracao.py                # carregamento do modelo e geração das manchetes
│   └── metricas.py               # ROUGE, BLEU, cópia, diversidade, memorização
├── scripts/
│   ├── coletar_noticias.py       # coleta pela API do Wikinotícias
│   ├── preparar_dados.py         # agrupamento por acontecimento e split
│   ├── treinar_modelo.py         # ajuste do T5 (Colab/T4)
│   └── avaliar_modelo.py         # avaliação e comparação base x ajustado
├── notebooks/treinar_colab.ipynb # passo a passo do treinamento no Colab
├── data/
│   ├── splits/                   # treino, validação e teste (versionados)
│   └── exemplos.json             # exemplos usados no playground
├── docs/                         # enunciado da prova
├── requirements.txt              # app e avaliação
├── requirements-data.txt         # coleta
├── requirements-train.txt        # treinamento no Colab
├── ESTADO_ATUAL.md               # decisões, comandos e próxima ação
└── ROTEIRO_VIDEO.md              # roteiro do vídeo de 5 minutos
```

## Ambiente e instalação

Requer Python 3.10 ou superior.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt        # app + avaliação
pip install -r requirements-data.txt   # coleta (só requests)
```

## Etapa 1 — Coleta

```bash
python scripts/coletar_noticias.py --quantidade 3000 --saida data/noticias.csv
```

| Argumento | Padrão | Descrição |
| --- | --- | --- |
| `--quantidade` | `1000` | notícias **aceitas** desejadas no CSV |
| `--saida` | `data/noticias.csv` | caminho do CSV de saída |
| `--max-requisicoes` | `400` | limite total de requisições à API |
| `--pausa` | `2.0` | pausa entre requisições, em segundos |
| `--timeout` | `30` | timeout de cada requisição, em segundos |

O script coleta em lotes de 20 páginas (`grnlimit=20`), salva o CSV ao final de
cada lote e retoma de onde parou. Notícias repetidas são descartadas por `pageid`
e por texto normalizado. Falhas transitórias usam recuo exponencial e respeitam o
cabeçalho `Retry-After`; se a API passar a limitar as requisições (HTTP 429) de
forma persistente, o script para e informa o total obtido.

**Resultado desta execução:** 3000 notícias aceitas, 140 duplicadas ignoradas e
285 descartadas, em 122 requisições.

## Etapa 2 — Dados

### Origem

API pública do Wikinotícias em português (`https://pt.wikinews.org/w/api.php`),
via `action=query` com `generator=random`, `grnnamespace=0`, `grnlimit=20`,
`prop=extracts`, `explaintext=1` e `exintro=1`. São páginas aleatórias do domínio
principal, e apenas a introdução de cada notícia é recuperada.

O Wikinotícias indica a licença Creative Commons Attribution 2.5 para o seu
conteúdo. **A atribuição e as condições de reutilização das notícias precisam ser
conferidas antes de publicar o dataset**, porque cada texto tem autoria própria.
Enquanto essa checagem não for feita, os dados são usados apenas para o trabalho
acadêmico e o repositório deve manter a referência à fonte e à licença.

### Colunas do CSV de coleta

| Coluna | Conteúdo |
| --- | --- |
| `id` | `pageid` da página no Wikinotícias |
| `titulo` | título completo da notícia (a manchete a ser gerada) |
| `texto_original` | texto da introdução exatamente como retornado pela API |
| `entrada` | texto preparado que alimenta o modelo |
| `url` | link da notícia (`https://pt.wikinews.org/?curid=<pageid>`) |
| `coletado_em` | data e hora da coleta em UTC (ISO 8601) |

### Critérios de preparação

O `texto_original` é preservado sem alterações. A coluna `entrada` é derivada dele:

1. Linhas de cabeçalho de data são removidas, inclusive no formato
   `Local • 14 de março de 2025`.
2. Linhas iniciais curtas de autoria ou agência são removidas, como `Agência
   Brasil`, `Agência VOA`, `Da redação` e `Por Marco Zero`. A remoção de linhas
   iniciadas por "Por" exige que a linha seja curta, sem pontuação de frase e que
   o restante pareça um nome próprio, para não descartar frases jornalísticas que
   apenas começam com "Por".
3. São usados no máximo os três primeiros parágrafos e no máximo 220 palavras.
4. São descartadas notícias com título vazio, entrada vazia ou com menos de 40
   palavras.
5. São descartadas notícias cujo título completo, normalizado, apareça
   literalmente na entrada. Esse filtro reduz a cópia direta do título pelo
   modelo, mas **não garante a qualidade dos pares**: restam casos de entradas
   truncadas no limite de palavras e de títulos muito próximos do texto.
6. Duplicatas são evitadas por `pageid` e pelo texto normalizado.

Os limites de 3 parágrafos, 220 palavras e 40 palavras mínimas são **escolhas
iniciais do experimento**, não requisitos do problema.

### Divisão por acontecimento

```bash
python scripts/preparar_dados.py
```

Antes de dividir, as notícias são agrupadas por acontecimento com duas regras:

- título normalizado idêntico; ou
- similaridade TF-IDF (palavras e bigramas) acima de **0,25**.

Os grupos inteiros vão para um único conjunto, com proporções 80/10/10. O script
falha se encontrar grupo, `id` ou texto repetido entre treino, validação e teste.

**Resultado:** 3000 notícias, 2827 grupos, maior grupo com 8 notícias;
2400 treino / 300 validação / 300 teste. Os três arquivos e o `resumo.json` ficam
em `data/splits/`.

Essa é uma heurística, não um agrupamento semântico perfeito. Duas notícias sobre
o mesmo evento com vocabulário muito diferente podem cair em grupos distintos, e
séries recorrentes (por exemplo, rodadas de campeonatos) formam grupos próprios.
O ganho principal é impedir que textos quase idênticos atravessem as divisões.

## Etapa 3 — Modelo escolhido

**`unicamp-dl/ptt5-base-portuguese-vocab`** — T5 *base* (220M parâmetros)
pré-treinado no corpus BrWaC com vocabulário português.

| Critério | Por que atende |
| --- | --- |
| Idioma | pré-treinamento só em português, sem desperdício de vocabulário multilíngue |
| Arquitetura | encoder-decoder, adequada para texto → manchete |
| T4 (16 GB) | ajuste com `batch 8 × grad_accum 2`, entrada de 512 tokens e fp16 |
| Inferência local | ~900 MB em fp32, roda em CPU nos 16 GB de RAM do notebook |

Alternativas consideradas: `ptt5-small` (60M, mais rápido porém mais fraco),
`ptt5-large` (740M, inviável para inferência confortável em CPU) e `mt5-base`
(tokenização pior para português). Modelos GPT-2 em português foram descartados
por serem apenas decodificadores, dificultando o controle texto → manchete.

O prefixo `manchete: ` é adicionado à entrada no treino e na inferência. O T5
original usa prefixos de tarefa; como o `ptt5` não foi pré-treinado com eles, o
prefixo funciona apenas como um marcador consistente de tarefa.

## Etapa 4 — Treinamento no Colab

Abra [notebooks/treinar_colab.ipynb](notebooks/treinar_colab.ipynb), selecione a
GPU T4 e execute as células. O comando central é:

```bash
python scripts/treinar_modelo.py \
    --modelo unicamp-dl/ptt5-base-portuguese-vocab \
    --dados data/splits \
    --saida modelos/manchetes-ptt5 \
    --epocas 6 --batch 8 --grad-accum 2 --lr 2e-4 \
    --fp16
```

Hiperparâmetros iniciais e o motivo:

| Parâmetro | Valor | Motivo |
| --- | --- | --- |
| épocas | 6 | 2400 exemplos é pouco; com parada antecipada (paciência 2) |
| batch efetivo | 16 (8 × 2) | cabe na T4 com 512 tokens de entrada |
| learning rate | 2e-4 | faixa usual para ajuste de T5 |
| warmup | 10% dos passos | evita desestabilizar o modelo nas primeiras atualizações |
| max entrada / saída | 512 / 64 tokens | entradas têm no máximo 220 palavras; títulos são curtos |
| seleção do checkpoint | ROUGE-L na validação | validação decide; teste é usado só no fim |

Saídas: `modelos/manchetes-ptt5/` (pesos e tokenizador) e
`metricas_treino.json` (hiperparâmetros, histórico, métricas de validação,
exemplos gerados e ambiente).

## Etapa 5 — Avaliação

```bash
# modelo original, sem ajuste
python scripts/avaliar_modelo.py \
    --modelo unicamp-dl/ptt5-base-portuguese-vocab --rotulo base

# modelo ajustado
python scripts/avaliar_modelo.py \
    --modelo modelos/manchetes-ptt5 --rotulo ajustado

# comparação
python scripts/avaliar_modelo.py --comparar avaliacao/base.json avaliacao/ajustado.json
```

Os dois comandos usam o mesmo conjunto de teste, o mesmo prefixo e a mesma
decodificação (3 sugestões, 8 feixes), o que torna a comparação justa.
As métricas são calculadas sobre a melhor das três sugestões de cada notícia.

| Métrica | O que mede |
| --- | --- |
| `rouge1`, `rouge2`, `rougeL` | proximidade lexical com o título original |
| `bleu` | precisão de n-gramas contra o título original |
| `jaccard_com_referencia` | sobreposição de vocabulário com o título original |
| `taxa_copia_exata` | fração de manchetes que aparecem literalmente no texto-fonte |
| `maior_trecho_comum` | maior sequência contígua copiada do texto-fonte |
| `aproveitamento_fonte` | fração de palavras da manchete que já existem no texto |
| `distinct1_entre_sugestoes` | diversidade entre as três sugestões |
| `taxa_igual_a_titulo_de_treino` | memorização de títulos vistos no treino |

`taxa_copia_exata` e `maior_trecho_comum` são as métricas de **originalidade**:
elas capturam o principal modo de falha de modelos de resumo/manchete, que é
devolver o próprio texto de entrada.

### Resultado do modelo original (300 notícias de teste)

Execução real, `avaliacao/base.json`:

| Métrica | Valor |
| --- | --- |
| `rouge1` | 0,2954 |
| `rouge2` | 0,1413 |
| `rougeL` | 0,2430 |
| `bleu` | 0,0460 |
| `taxa_copia_exata` | 0,3485 |
| `aproveitamento_fonte` | 0,9813 |
| `maior_trecho_comum` | 0,8457 |
| `distinct1_entre_sugestoes` | 0,3541 |
| `palavras_por_manchete` | 33,21 |

O `ptt5` **sem ajuste não gera manchetes**: ele continua o texto recebido. Três
evidências nos números acima:

- 34,85% das sugestões aparecem **literalmente** no texto de entrada;
- 98,13% das palavras das sugestões já existem no texto-fonte;
- as sugestões têm 33 palavras em média, o tamanho de uma frase, não de manchete.

A pontuação de ROUGE (0,2954) existe só porque as frases copiadas compartilham
palavras com o título real. É um exemplo claro de métrica lexical enganosa: sem
as métricas de originalidade, o modelo base pareceria competitivo. Isso é
esperado, porque o `ptt5` não passou por instrução e não conhece a tarefa.

É por isso que a comparação com o modelo ajustado é a evidência central do
trabalho: ROUGE sozinho não distingue um gerador de manchetes de um copiador.

### Limitações conhecidas

- 3000 notícias é um conjunto pequeno para geração abstrativa; o modelo tende a
  reaproveitar trechos da entrada em vez de parafrasear.
- O agrupamento por acontecimento é heurístico.
- Títulos de séries (rodadas de campeonato, apurações eleitorais) têm formato
  repetitivo e podem ser reproduzidos mecanicamente.
- A avaliação automática não mede factualidade nem adequação jornalística.
- O Wikinotícias tem cobertura irregular por período; a distribuição por ano é
  desigual.

## Etapa 6 — Playground Streamlit

```bash
streamlit run app.py
```

O app carrega o modelo de `MODELO_ID` (variável de ambiente ou *secret* do
Streamlit) e, se ela não existir, usa `modelos/manchetes-ptt5` ou, como último
recurso, o modelo base.

Recursos: entrada de texto livre, exemplos prontos, escolha de quantidade de
sugestões, feixes de busca, limite de tokens e amostragem com temperatura. A
barra lateral mostra o modelo em uso, as métricas do conjunto de teste e as
limitações.

### Deploy

Recomendado: **Hugging Face Spaces** com SDK Streamlit (16 GB de RAM em CPU),
porque o modelo de 220M não cabe com folga no plano gratuito do Streamlit
Community Cloud.

1. Publique o modelo ajustado no Hub (`scripts/treinar_modelo.py --push-hub ...`
   ou a célula 9 do notebook).
2. Crie um Space novo do tipo Streamlit e envie `app.py`, `manchetes/`,
   `requirements.txt` e `data/exemplos.json`.
3. Defina o *secret* `MODELO_ID=<usuario>/manchetes-ptt5`.
4. Anote a URL pública para o TXT de entrega.

Alternativa: Streamlit Community Cloud apontando para o repositório do GitHub,
com `MODELO_ID` em *Secrets* e um modelo menor se faltar memória.

## Entrega

O arquivo `entrega.txt` deve conter:

- URL do repositório no GitHub;
- URL do playground publicado;
- URL do vídeo com pelo menos 5 minutos.

O roteiro do vídeo está em [ROTEIRO_VIDEO.md](ROTEIRO_VIDEO.md).
