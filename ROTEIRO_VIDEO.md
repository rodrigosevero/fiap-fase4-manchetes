# Roteiro do vídeo (5 a 7 minutos)

Objetivo: explicar a estratégia do modelo, mostrar o ajuste com Hugging Face
Transformers, a avaliação e o playground funcionando.

Grave a tela com o VS Code aberto no projeto e o navegador com o playground
publicado. Fale em português, sem pressa, e mostre números reais.

## 0:00 — Abertura (30 s)

- Quem você é e qual é o desafio.
- Frase de efeito: "o objetivo é gerar três sugestões de manchete em português a
  partir do texto de uma notícia, com um T5 ajustado".

## 0:30 — Tema e dados (60 s)

- Tema: geração de manchetes jornalísticas em português.
- Fonte: API pública do Wikinotícias (`generator=random`, só a introdução).
- Mostre `scripts/coletar_noticias.py` e comente os critérios:
  remoção de cabeçalho de data e créditos, até 3 parágrafos / 220 palavras,
  descarte de menos de 40 palavras e de títulos que aparecem na entrada.
- Diga o número real: 3000 notícias aceitas, 140 duplicadas, 285 descartadas.
- **Diga que a licença e a atribuição das notícias precisam ser conferidas antes
  de publicar o dataset.**

## 1:30 — Divisão por acontecimento (60 s)

- Abra `data/splits/resumo.json`.
- Explique: agrupamento por título idêntico e similaridade TF-IDF ≥ 0,25;
  grupos inteiros vão para um só conjunto; 2400/300/300.
- Por que importa: quase-duplicatas do mesmo evento vazariam a resposta e
  inflariam as métricas.

## 2:30 — Modelo e ajuste (90 s)

- Modelo: `unicamp-dl/ptt5-base-portuguese-vocab`, 220M, T5 encoder-decoder,
  pré-treinado no BrWaC.
- Justificativa: só português, cabe na T4, roda em CPU local.
- Mostre `scripts/treinar_modelo.py` e o comando do Colab:
  `--epocas 6 --batch 8 --grad-accum 2 --lr 2e-4 --fp16`.
- Comente a seleção do checkpoint por ROUGE-L na validação e a parada antecipada.
- Mostre uma captura do treino no Colab (loss e ROUGE por época).

## 4:00 — Avaliação base x ajustado (90 s)

- Mostre a tabela de `avaliacao/comparacao.json`.
- Explique que os dois rodaram no mesmo teste, mesmo prefixo e mesma decodificação.
- Leia os números reais: ROUGE-1, ROUGE-L, BLEU, `taxa_copia_exata`,
  `maior_trecho_comum`, `distinct1`.
- Ponto-chave: **o modelo base não gera manchetes, ele copia o texto** — por isso
  a métrica de originalidade é necessária, não basta ROUGE.
- Mostre um exemplo lado a lado (base x ajustado x título real).

## 5:30 — Falhas e limitações (30 s)

- Cite 2 falhas concretas que você viu: título repetitivo de série esportiva,
  manchete truncada, nome próprio errado ou detalhe inventado.
- Reconheça: 3000 notícias é pouco para geração abstrativa; a avaliação automática
  não mede factualidade.

## 6:00 — Playground (60 s)

- Abra o link público do Streamlit.
- Cole uma notícia, gere as três sugestões, mexa nos feixes e no limite de tokens.
- Mostre a barra lateral com as métricas e as limitações.

## 7:00 — Fechamento (20 s)

- Recapitule: dados → split por acontecimento → ajuste do T5 → avaliação com
  originalidade → Streamlit.
- Diga as URLs: repositório no GitHub, playground, e que o `entrega.txt` está no
  repositório.

## Checklist antes de publicar o vídeo

- [ ] Duração ≥ 5 minutos (ideal 6 a 7).
- [ ] Números do vídeo batem com `avaliacao/comparacao.json`.
- [ ] O playground abre e gera sugestões durante a gravação.
- [ ] Áudio audível e tela legível (aumentar a fonte do editor).
- [ ] Vídeo publicado como "não listado" no YouTube e link copiado.
