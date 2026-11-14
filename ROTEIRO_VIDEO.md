# Roteiro do vídeo (7 minutos)

Objetivo: explicar a estratégia, mostrar o ajuste com Hugging Face Transformers, a
avaliação e o playground publicado. Todos os números abaixo são reais e estão em
`avaliacao/comparacao.json`.

**Antes de gravar:** abra o VS Code no projeto, o app publicado em
<https://fiap-fase4-manchetes-xjjk25vdjk25m5pdcuwzjl.streamlit.app/> e o
`comparacao.json`. Aumente a fonte do editor. Feche abas que não vai usar.

---

## 0:00 — Abertura (30 s)

> "Olá, sou o Rodrigo. Este é o projeto da prova substitutiva da Fase 4 de Machine
> Learning Engineering. O desafio é gerar conteúdo original com Hugging Face
> Transformers; eu escolhi gerar **três sugestões de manchete em português** a
> partir do texto de uma notícia."

Mostre a tela do app já aberto.

## 0:30 — Tema e dados (60 s)

Abra `scripts/coletar_noticias.py` e leia só os critérios:

> "Coletei 3000 notícias da API pública do Wikinotícias. O pré-processamento
> remove cabeçalho de data e créditos de agência, usa até três parágrafos e no
> máximo 220 palavras, descarta textos com menos de 40 palavras e descarta
> exemplos cujo título aparece literalmente no texto."

> "Resultado da coleta: 3000 notícias aceitas, 140 duplicadas e 285 descartadas."

**Diga a ressalva de licença:**

> "As notícias vêm do Wikinotícias sob CC BY 2.5. A atribuição e as condições de
> reutilização precisam ser conferidas antes de publicar o dataset."

## 1:30 — Divisão por acontecimento (60 s)

Abra `data/splits/resumo.json`.

> "O ponto mais delicado é separar treino, validação e teste. Se eu dividisse por
> notícia, duas matérias sobre o mesmo evento cairiam em lados diferentes e o
> modelo 'acertaria' por já ter visto aquele acontecimento."

> "Então agrupei por título idêntico **e** por similaridade TF-IDF acima de 0,25,
> e movi grupos inteiros. Deu 2827 grupos, divididos em 2400 de treino, 300 de
> validação e 300 de teste. O script falha se encontrar grupo, id ou texto
> repetido entre os conjuntos."

## 2:30 — Modelo e ajuste (90 s)

Abra `scripts/treinar_modelo.py`.

> "O modelo é o `unicamp-dl/ptt5-base-portuguese-vocab`: um T5 de 220 milhões de
> parâmetros pré-treinado só em português. Escolhi ele porque é encoder-decoder,
> que é o formato certo para texto → manchete, cabe numa T4 e roda em CPU no meu
> notebook de 16 GB."

> "O ajuste usou seis épocas, batch 8 com acumulação de 2, learning rate 2e-4 e
> fp16. O melhor checkpoint é escolhido por ROUGE-L **na validação**, e o teste só
> é usado no fim. Levou 15 minutos numa T4."

Mostre o notebook do Colab ou o `metricas_treino.json`.

## 4:00 — Avaliação base × ajustado (90 s)

Abra a tabela de `avaliacao/comparacao.json`.

> "Rodei o modelo original e o ajustado no mesmo conjunto de teste, com o mesmo
> prefixo e a mesma decodificação: três sugestões, oito feixes."

> "ROUGE-1 subiu de 0,29 para 0,46. ROUGE-L de 0,24 para 0,42. BLEU quadruplicou,
> de 0,046 para 0,163."

**Este é o ponto principal da explicação:**

> "Mas a métrica que eu considero mais importante é a de originalidade. O modelo
> original copiava o texto: 34,8% das sugestões apareciam literalmente na entrada,
> e o maior trecho copiado era 85% da manchete. Depois do ajuste, a cópia caiu
> para 1,8% e o maior trecho para 48%. E as manchetes passaram de 33 para 12
> palavras — saíram do tamanho de frase para o tamanho de manchete."

> "Se eu olhasse só ROUGE, o modelo original pareceria competitivo. Ele não é: ele
> só devolve o texto de entrada."

Mostre um exemplo lado a lado:

- Referência: *"Após 45 anos, Fidel Castro entra numa igreja"*
- Base: *"A morte do papa João Paulo II fez com que o presidente de Cuba Fidel Castro entrasse numa igreja..."*
- Ajustado: *"Sócrates diz que vai trabalhar para que referendo português aconteça em outubro"* (outro exemplo)

## 5:30 — Falhas e limitações (45 s)

> "Três falhas que eu documentei. Primeira, invenção factual: para a notícia do
> Fidel Castro, o modelo escreveu 'Morre o papa João Paulo II, presidente de
> Cuba' — juntou duas entidades do texto de um jeito que não aconteceu."

> "Segunda, formato: 2 de 300 notícias geraram manchetes longas, de até 41
> palavras. Terceira, cópia residual: 16 de 900 sugestões ainda são transcrição
> literal."

> "Nenhuma métrica automática pega a invenção factual. Por isso o app avisa que a
> revisão humana é obrigatória."

## 6:15 — Playground (45 s)

Abra <https://fiap-fase4-manchetes-xjjk25vdjk25m5pdcuwzjl.streamlit.app/>.

> "O playground está publicado no Streamlit Community Cloud. Colo uma notícia,
> clico em gerar e recebo três sugestões."

Cole um texto, gere, e mostre na barra lateral:

> "A barra lateral mostra o modelo que está carregado — `rsevero/manchetes-ptt5`,
> do Hugging Face Hub — e a tabela de métricas do teste."

Ajuste um controle (feixes ou máximo de tokens) e gere de novo rapidamente.

## 7:00 — Fechamento (20 s)

> "Resumindo: dados do Wikinotícias, divisão por acontecimento, ajuste do T5 com
> Transformers, avaliação que olha qualidade **e** originalidade, e um playground
> publicado. Repositório, app e este vídeo estão no arquivo de entrega."

---

## Checklist antes de publicar

- [ ] Duração ≥ 5 minutos (ideal 6–7).
- [ ] Todos os números ditos batem com `avaliacao/comparacao.json`.
- [ ] O app publicado abriu e gerou sugestões durante a gravação.
- [ ] O nome do modelo e a URL aparecem legíveis na tela.
- [ ] Áudio audível, sem notificações no meio.
- [ ] Vídeo publicado no YouTube como "não listado" e link copiado para `entrega.txt`.
