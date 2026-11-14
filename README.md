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

> **Status:** modelo ajustado, avaliado e publicado em
> <https://huggingface.co/rsevero/manchetes-ptt5>. Resultados reais na seção
> [Resultados](#resultados-300-notícias-de-teste). Playground e vídeo em
> finalização — acompanhamento em [ESTADO_ATUAL.md](ESTADO_ATUAL.md).

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

### Resultados (300 notícias de teste)

Treino: 2400 exemplos, 750 passos, 6 épocas, 15,3 min numa Tesla T4
(Python 3.13, torch 2.14.1+cu130, transformers 5.22.1).
Seleção do checkpoint por ROUGE-L na validação (300 notícias):
`rouge1` 0,4303 · `rouge2` 0,2526 · `rougeL` 0,3906 · `bleu` 0,1502 · loss 2,0835.

| Métrica | Base | Ajustado | Variação |
| --- | --- | --- | --- |
| `rouge1` | 0,2954 | **0,4574** | +55% |
| `rouge2` | 0,1413 | **0,2764** | +96% |
| `rougeL` | 0,2430 | **0,4209** | +73% |
| `bleu` | 0,0460 | **0,1634** | +255% |
| `jaccard_com_referencia` | 0,1777 | **0,2731** | +54% |
| `taxa_copia_exata` | 0,3481 | **0,0178** | −95% |
| `aproveitamento_fonte` | 0,9813 | **0,8967** | −9% |
| `maior_trecho_comum` | 0,8456 | **0,4812** | −43% |
| `distinct1_entre_sugestoes` | 0,3536 | **0,4171** | +18% |
| `palavras_por_manchete` | 33,21 | **12,44** | −63% |
| `segundos_por_noticia` | 9,14 | **3,54** | −61% |
| `taxa_igual_a_titulo_de_treino` | 0,0000 | **0,0000** | — |

Três leituras que sustentam a conclusão:

1. **Qualidade:** ROUGE-1 e ROUGE-L sobem 55% e 73%, e o BLEU quadruplica. As
   manchetes passam de 33 para 12,4 palavras — saem do tamanho de frase e chegam
   ao tamanho de manchete.
2. **Originalidade:** a cópia literal cai de 34,8% para 1,8% das sugestões e o
   maior trecho copiado cai de 85% para 48% da manchete. O modelo deixou de
   devolver o texto de entrada.
3. **Memorização:** nenhuma sugestão coincide com títulos do treino
   (`taxa_igual_a_titulo_de_treino` = 0), então o ganho não vem de decorar.

O `aproveitamento_fonte` continua alto (0,90) porque manchetes de notícia
reaproveitam nomes próprios e termos do texto. Isso é esperado e não indica cópia:
`maior_trecho_comum` mostra que as sequências longas copiadas caíram pela metade.

### Exemplos reais

**Acerto** — referência: *"Sócrates disse que trabalha para que referendo sobre a
Constituição Européia aconteça em Outubro, em Portugal"*

```
ajustado: Sócrates diz que vai trabalhar para que referendo português aconteça em outubro
          Primeiro-ministro de Portugal diz que vai trabalhar para que referendo aconteça em outubro
```

**Acerto** — referência: *"Lula disse ontem que vai 'cortar da própria carne'"*

```
ajustado: Lula diz que vai tomar todas as medidas necessárias para combater a corrupção
          Lula diz que não vai encobrir nenhum caso de corrupção
```

**Falha de factualidade** — referência: *"Após 45 anos, Fidel Castro entra numa igreja"*

```
base    : A morte do papa João Paulo II fez com que o presidente de Cuba Fidel Castro entrasse numa igreja...
ajustado: Morre o papa João Paulo II, presidente de Cuba
```

O ajustado criou uma manchete plausível na forma, mas **errada no conteúdo**:
confundiu o papa com o presidente de Cuba. É o risco central de geração abstrativa
e o motivo de a revisão humana ser obrigatória.

**Falha de formato** — 2 de 300 notícias geraram manchetes com mais de 28 palavras
(a maior com 41), voltando ao comportamento de frase corrida.

**Falha residual de cópia** — 16 de 900 sugestões (1,8%) aparecem literalmente no
texto, como *"Boa Vista, capital de Roraima, amanheceu diferente do normal"* para a
referência *"Propaganda eleitoral suja Boa Vista"*. Algumas dessas cópias até leem
bem como manchete, mas não são geração: são transcrição.

### Limitações conhecidas

- **Invenção factual.** O modelo combina entidades do texto de forma plausível e
  errada, como no caso "Morre o papa João Paulo II, presidente de Cuba". Nenhuma
  métrica automática aqui detecta isso; é preciso revisão humana.
- **Formato instável em casos raros.** 2 de 300 notícias produziram manchetes
  longas (até 41 palavras), voltando ao estilo de frase corrida.
- **Cópia residual.** 1,8% das sugestões são transcrição literal de um trecho.
  A métrica `taxa_copia_exata` também pode marcar frases legítimas que por acaso
  existem no texto, então o número é um limite superior.
- **Conjunto pequeno.** 3000 notícias é pouco para geração abstrativa; mais dados
  devem reduzir a cópia e melhorar a paráfrase.
- **Agrupamento heurístico.** O agrupamento por acontecimento usa TF-IDF e pode
  não juntar notícias do mesmo evento escritas com vocabulário muito diferente.
- **Padrões de série.** Rodadas de campeonato e apurações eleitorais têm título
  repetitivo e podem ser reproduzidas mecanicamente.
- **Cobertura desigual.** O Wikinotícias tem distribuição irregular por ano, e o
  modelo não conhece eventos posteriores ao corpus.
- **Avaliação lexical.** ROUGE e BLEU medem sobreposição de palavras, não
  factualidade, clareza ou adequação jornalística.

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

**Aplicação publicada:**
<https://fiap-fase4-manchetes-xjjk25vdjk25m5pdcuwzjl.streamlit.app/>

Plataforma: **Streamlit Community Cloud**, que constrói direto do repositório do
GitHub. O modelo é baixado do Hugging Face Hub na primeira geração.

Como reproduzir o deploy:

1. Publique o modelo: `python scripts/treinar_modelo.py --push-hub SEU_USUARIO/manchetes-ptt5`
   (ou use a célula 9 de [notebooks/treinar_colab.ipynb](notebooks/treinar_colab.ipynb)).
2. Em <https://share.streamlit.io>, escolha **Deploy a public app from GitHub**.
3. Repository `SEU_USUARIO/fiap-fase4-manchetes`, branch `main`, main file `app.py`.
4. Em **Advanced settings**, Python 3.12 e o secret:

   ```toml
   MODELO_ID = "SEU_USUARIO/manchetes-ptt5"
   ```

5. **Deploy**. O `requirements.txt` já usa o índice CPU do PyTorch, o que reduz o
   build de ~2 GB de downloads para ~200 MB.

**Verificação feita na aplicação publicada:** as três sugestões geradas para um
texto inédito foram coerentes com a notícia, e a barra lateral exibiu o modelo
`rsevero/manchetes-ptt5` e a tabela de métricas do teste.

> **Por que não Hugging Face Spaces:** a conta usada no projeto recebeu
> `Quota exceeded for flavor cpu-basic (limit=0)`, ou seja, sem direito a
> instância gratuita de CPU. O Space chegou a ser criado com os arquivos
> enviados, mas não entra em execução. Por isso o deploy foi feito no Streamlit
> Community Cloud.

## Entrega

Repositório: <https://github.com/rodrigosevero/fiap-fase4-manchetes>
Modelo no Hub: <https://huggingface.co/rsevero/manchetes-ptt5>

O arquivo [`entrega.txt`](entrega.txt) deve conter as três URLs exigidas:

- URL do repositório no GitHub;
- URL do playground publicado;
- URL do vídeo com pelo menos 5 minutos.

O roteiro do vídeo está em [ROTEIRO_VIDEO.md](ROTEIRO_VIDEO.md).
