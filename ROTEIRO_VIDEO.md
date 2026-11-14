# Roteiro do vídeo — versão curta

Duração alvo: **6 a 7 minutos**. Fale devagar, sem pressa. Os textos abaixo são
sugestões: diga com suas palavras, só não pule os números.

## Antes de gravar

- Abra o app: <https://fiap-fase4-manchetes-xjjk25vdjk25m5pdcuwzjl.streamlit.app/>
- Abra o VS Code no projeto, com fonte maior.
- Deixe um texto de notícia copiado para colar no app.

## Como falar os termos

| Escrito | Fale |
| --- | --- |
| T5 | "tê cinco" |
| T4 | "tê quatro" |
| ROUGE | "ruje" |
| BLEU | "blêu" |
| Streamlit | "strímlit" |
| Hugging Face | "râguin feice" |
| Transformer | "transfórmer" |
| Wikinotícias | "wikinotícias" |

> Não precisa falar nome de modelo com hífen. Diga só **"um modelo T5 em
> português, de 220 milhões de parâmetros"**.

---

## 0:00 — Abertura (30 s)

**Mostre:** o app já aberto.

**Fale (2 frases):**
- "Olá, sou o Rodrigo. Este é o projeto da prova substitutiva da Fase 4 de Machine Learning Engineering."
- "O desafio é gerar texto com Transformers. Eu escolhi gerar três sugestões de manchete em português a partir do texto de uma notícia."

## 0:30 — Os dados (60 s)

**Mostre:** `scripts/coletar_noticias.py`.

**Fale (3 frases):**
- "Peguei 3000 notícias do Wikinotícias, que é uma fonte pública."
- "Antes de usar, limpei o texto: tirei a data e o nome da agência no começo, e cortei em três parágrafos ou 220 palavras."
- "Também descartei notícias curtas demais e notícias cujo título já aparecia no texto, para o modelo não aprender a copiar."

**Um número:** "No total, 3000 aceitas, 140 repetidas e 285 descartadas."

**Importante falar:** "As notícias são de licença livre, mas os créditos e as regras de uso precisam ser conferidos antes de publicar os dados."

## 1:30 — A divisão dos dados (60 s)

**Mostre:** `data/splits/resumo.json`.

**Fale (3 frases):**
- "Separei os dados em treino, validação e teste. O cuidado principal foi não deixar duas notícias do mesmo evento em lados diferentes."
- "Se isso acontecesse, o modelo poderia acertar só porque já tinha visto aquele assunto."
- "Então juntei as notícias parecidas e movi o grupo inteiro para o mesmo lado."

**Um número:** "Ficou 2400 para treinar, 300 para validar e 300 para testar."

## 2:30 — O modelo e o treino (90 s)

**Mostre:** `scripts/treinar_modelo.py`.

**Fale (3 frases):**
- "Usei um modelo T5 em português, de 220 milhões de parâmetros. Ele lê um texto e escreve outro texto, que é exatamente o que eu preciso."
- "Escolhi esse tamanho porque treina numa placa T4 do Colab e também roda no meu notebook, que não tem placa de vídeo."
- "O treino levou 15 minutos, com 6 épocas. Quem escolhe o melhor resultado é a validação; o teste fica guardado para o final."

## 4:00 — O resultado (90 s)

**Mostre:** `avaliacao/comparacao.json`.

**Fale (4 frases):**
- "Rodei o modelo original e o modelo treinado no mesmo teste, do mesmo jeito."
- "A nota de semelhança com o título real subiu de 0,29 para 0,46."
- "Mas o número que mais importa é outro: o modelo original copiava o texto. Quase 35 por cento das sugestões apareciam iguais no texto de entrada."
- "Depois do treino, a cópia caiu para menos de 2 por cento. E as manchetes passaram de 33 para 12 palavras, ou seja, ficaram com cara de manchete e não de parágrafo."

**Conclusão (fale com calma):** "Se eu olhasse só a nota de semelhança, o modelo sem treino pareceria bom. Ele não é: ele só devolve o texto que recebeu."

## 5:30 — Uma falha real (30 s)

**Mostre:** o exemplo no README.

**Fale (2 frases):**
- "Nem tudo deu certo. Numa notícia sobre o Fidel Castro, o modelo escreveu: 'morre o papa João Paulo Segundo, presidente de Cuba'."
- "Ele misturou duas informações do texto e criou algo que não aconteceu. Por isso o app avisa que precisa de revisão humana."

## 6:00 — O app (60 s)

**Mostre:** o app publicado. Cole a notícia e clique em gerar.

**Fale (2 frases):**
- "Aqui está o app publicado. Eu colo a notícia e ele me dá três sugestões."
- "Na lateral aparecem o modelo que está carregado e as notas do teste."

**Ação:** gere as sugestões e leia uma em voz alta.

## 7:00 — Fechamento (20 s)

**Fale (2 frases):**
- "Resumindo: dados públicos, divisão por evento, treino do T5 com Transformers, uma avaliação que olha cópia e qualidade, e o app publicado."
- "O repositório, o app e este vídeo estão no arquivo de entrega."

---

## Checklist

- [ ] Mais de 5 minutos.
- [ ] Disse os números: 3000 notícias, 0,29 → 0,46 e 35% → 2% de cópia.
- [ ] O app abriu e gerou as sugestões na gravação.
- [ ] Sem notificações na tela.
- [ ] Publiquei no YouTube como "não listado" e copiei o link.
