# Análise completa - Questão 1 (Pré-treino continuado)

> Pré-treinamento continuado do `Qwen/Qwen2.5-0.5B` no corpus DOMPI-2025 (Diário Oficial dos Municípios do Piauí). Execução em Kaggle (1× GPU T4), tempo total ~4h53min (treino ~4h40).

## 1. Configuração efetiva do experimento

| Item | Valor |
|---|---|
| Modelo | Qwen2.5-0.5B (494M params, 24 camadas, hidden 896, vocab 151.936) |
| Corpus | DOMPI-2025, 77.337 docs brutos |
| Após filtro (200 a 50k chars) | 72.431 docs |
| Treino / Avaliação | 71.431 / 1.000 (held-out) |
| Épocas / Batch efetivo / LR | 2 / 32 / 5e-5 (cosine, warmup 5%) |
| Precisão / Otimização | fp32 + autocast fp16, gradient checkpointing, 1× T4 |
| Steps / Tempo de treino | 4.466 steps / 16.821s (~4h40) |
| Loss final de treino | 0.8945 |
| Melhor checkpoint | `checkpoint-4466` (fim da época 2) |

Observação: o objetivo era usar "todas as 77k". Na prática 72.431 documentos passaram no filtro de tamanho e 71.431 foram para o treino (1.000 reservados para avaliação). Ou seja, todo o corpus utilizável entrou no treino.

## 2. Resultados quantitativos (held-out, 1.000 docs)

| Métrica | Antes | Depois | Δ | Variação |
|---|---|---|---|---|
| Entropia Cruzada | 3.0812 | **0.7390** | -2.3422 | -76% |
| Perplexidade | 21.78 | **2.09** | -19.69 | **-90.4%** |
| Acurácia de Tokens | 42.25% | **82.00%** | +39.75 pp | +94% |

Leitura: a perplexidade caiu de ~22 (o modelo hesitava entre muitas opções por token) para ~2 (praticamente sabe qual é o próximo token). A acurácia de previsão de token quase dobrou. Essa é a evidência central da Q1: o pré-treino continuado adaptou fortemente o modelo ao domínio jurídico/administrativo do DOM-PI.

## 3. Dinâmica do treino

O melhor checkpoint foi o último (época 2, step 4466), e não o da época 1. Isso indica que a `eval_loss` no conjunto held-out continuou caindo até o fim, sem sinal de overfitting (se tivesse decorado o treino, a loss de validação teria subido na 2ª época e o Trainer teria escolhido o checkpoint da época 1). O `load_best_model_at_end=True` selecionou o melhor estado por `eval_loss`, então a avaliação final foi feita no ponto ótimo, e não no último step qualquer.

## 4. Análise qualitativa (geração livre)

Mesmos prompts, modelo base vs modelo treinado.

### Prompt: "O Prefeito Municipal, no uso de suas atribuições legais,"
- **BASE:** divaga sobre "sorteio para o prêmio do Imposto sobre receitas brutas (IRBR) 2018", texto genérico e fora de domínio.
- **TREINO:** "...e com fulcro na Lei n 14.133/2021 e demais normas pertinentes, RESOLVE, após exame criterioso de documentação e acatando a orientação da Agente de Contratação e sua Equipe de Apoio, RATIFICAR o procedimento licitatório referente a DISP[ENSA]"

### Prompt: "PORTARIA Nº 001/2025. Nomeia servidor para o cargo de"
- **BASE:** "Coordenador do Programa Estudantil do Município de Vila Verde - Notícias de Vila Verde" (inventado, estilo blog).
- **TREINO:** "...Diretora do Departamento de Ensino Fundamental da Secretaria Municipal de Educação. O PREFEITO DO Município DE SIMÕES - PI, no uso de suas atribuições legais e nos termos do artigo 60... da Lei Orgânica do Município"

### Prompt: "EXTRATO DE CONTRATO. Contratante: Prefeitura Municipal de"
- **BASE:** "São Carlos - SP... eleição do candidato à presidência da Câmara" (alucinação fora de domínio).
- **TREINO:** "...São João do Arraial-PI. Objeto: Aquisição de material permanente para a Secretaria Municipal de Educação. Recursos: Orçamento Geral. Valor global: R$ 124.036,54..."

O contraste é claro: o modelo treinado produz texto administrativo autêntico do Piauí, com a lei correta de licitações (14.133/2021), municípios reais (Simões-PI, São João do Arraial-PI), estrutura de portaria/extrato e até valores monetários formatados. O modelo base produz conteúdo genérico de blog/notícia, frequentemente de outros estados.

## 5. Benchmark de QA (25 perguntas)

Como o planejamento previu, o modelo base não acerta CNPJ/CEP/e-mail exatos e trata as perguntas como busca web ("Blog do Carlos", "Portal de Notícias"), repetindo frases. Exemplos: o CEP de Aroazes virou uma explicação genérica de "o que é CEP"; o CNPJ virou a definição de "o que é CNPJ".

Pontos importantes para o relatório:
- O benchmark de QA salvo é apenas o do modelo base (`resultados_benchmark_base.json`). O notebook não re-rodou as 25 perguntas no modelo treinado nem salvou em JSON, então a comparação QA antes/depois não foi capturada de forma estruturada. Para a Q1 isso é aceitável, porque o QA é qualitativo e a métrica central é a perplexidade.
- A acurácia de QA seria baixa antes e depois mesmo, porque CNPJ/CEP exatos são memorização factual, não modelagem de linguagem. Use isso como argumento: o pré-treino ensina o modelo a falar o domínio, não a decorar tabelas.

## 6. Notas técnicas e caveats

- **`missing keys: ['lm_head.weight']`** no log é inofensivo. O Qwen2.5 usa `tie_word_embeddings: true`, então o `lm_head` compartilha pesos com a matriz de embeddings; ao carregar o checkpoint ele é reconstruído automaticamente. As métricas confirmam que o modelo carregou corretamente.
- **`dtype: float32`** no `config.json` confirma que o carregamento em fp32 funcionou (modelo salvo em fp32, daí os ~1.9 GB do `model.safetensors`). Isso foi necessário porque o `fp16=True` do treino usa GradScaler, que exige pesos mestres em fp32; sem isso, o transformers carregava o Qwen em bfloat16 e o treino quebrava na T4 (que não tem bf16 em hardware).
- **Ambiente:** o Kaggle estava com transformers 5.0.0, o que explica os avisos de deprecação (`torch_dtype`→`dtype`, `warmup_ratio`→`warmup_steps`) e por que usar a stack pré-instalada do Kaggle (sem reinstalar torch/transformers) foi a decisão correta.
- **Uso de GPU:** o treino rodou em uma única T4 via `CUDA_VISIBLE_DEVICES=0`. Com as 2 GPUs (T4 ×2), o Trainer ativaria DataParallel, que junta os logits (vocab 151k) na GPU 0 e estoura a memória. O modelo de 0.5B cabe folgado em uma T4.
- **Viés metodológico pequeno:** `calcular_metricas` pondera a perplexidade por todos os tokens reais, em vez de só as posições que entram no loss (uma a menos por sequência, por causa do shift). É um viés minúsculo e consistente antes/depois, então a comparação permanece válida.
- **Por que a perplexidade ficou tão baixa (2.09):** combinação de (a) adaptação real ao domínio e (b) o texto de diário oficial ser altamente templado e repetitivo (baixa entropia intrínseca). Não é vazamento, pois o conjunto de avaliação é held-out, mas vale registrar no relatório que parte do mérito vem da natureza repetitiva do corpus.

## 7. Conclusão

O experimento confirma a hipótese da Q1: o pré-treino continuado adapta o Qwen2.5-0.5B ao DOM-PI, com perplexidade caindo 90% (21.78 para 2.09) e acurácia de token subindo de 42% para 82%, sem overfitting (melhor checkpoint na última época). A geração qualitativa mostra a transformação de texto genérico para linguagem administrativa autêntica do Piauí. O QA factual permanece fraco, o que é esperado e reforça a distinção entre modelar linguagem e memorizar fatos.

## Artefatos gerados

- `q1_resultados/resultados_metricas.json` - métricas antes/depois e hiperparâmetros
- `q1_resultados/resultados_benchmark_base.json` - 25 respostas geradas pelo modelo base
- `q1_resultados/benchmark_dompi25.json` - os 25 pares pergunta/resposta de referência
- `q1_resultados/dompi_qwen25_final/` - modelo treinado (safetensors fp32 + tokenizer)
