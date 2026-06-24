# Tempo estimado para rodar as questões restantes

> Análise cruzando PLANO-3, PLANO-4, TRABALHO FINAL.md e as configs reais de treino dos notebooks Q2 e Q3. Q1 já rodada (ver `analise-questao-01.md`).

## Ponto central

Só existem notebooks para Q1, Q2 e Q3. As questões 4 (destilação), 5 (RAG) e 6 (guardrails) do `TRABALHO FINAL.md` **não têm código ainda**. Então "tempo pra rodar" só se aplica de fato a Q2 e Q3; Q4 a Q6 são "tempo pra construir do zero + rodar".

## Calibração (do quanto o Q1 realmente rodou)

O Q1 treinou a ~8,5 sequências/s na T4 (full SFT, grad checkpointing, len 256). Q2 e Q3 usam o `docentesDC`, que é de 5 a 35x menor que os 71k docs do Q1, então o tempo de GPU cai pra dezenas de minutos e o que domina passa a ser o overhead fixo (carregar modelo, baseline, geração, salvar): ~10 a 15 min por notebook.

## Tabela de tempo estimado (T4 Kaggle)

| Q     | O que é                                | Status do código                                  | Tempo de GPU (rodar)                                                                                | Prep/correção antes de rodar                                                                             | Total realista             |
| ----- | -------------------------------------- | ------------------------------------------------- | --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | -------------------------- |
| **2** | Full SFT, Qwen2.5-0.5B                 | Notebook existe, precisa das correções do PLANO-3 | ~15 a 40 min (se capar em ~2.000 pares); ~1h30 a 2h se usar todos os ~13k registros                 | ~45 a 90 min (trocar fonte pro HF, matar lógica de `topico_inferido`, máscara de loss, perguntas reais)  | **~1h30 a 2h30**           |
| **3** | LoRA/QLoRA, Qwen2.5-0.5B               | Notebook existe, precisa das correções do PLANO-4 | ~25 a 40 min (0.5B) **+ opcional ~45 a 60 min** (bônus QLoRA 1.5B)                                  | ~20 a 40 min (bug do `enable_input_require_grads`, subir artefatos da Q2, rodar comparação Full vs LoRA) | **~1h a 2h30** (com bônus) |
| **4** | Destilação (teacher para student)      | **Não existe**                                    | ~1h a 2h (gerar dataset sintético + treinar student + benchmark de 100Q nos 2 modelos antes/depois) | ~3h a 5h (construir tudo do zero)                                                                        | **~5h a 7h**               |
| **5** | RAG (Standard/Agentic/Self-Reflective) | **Não existe**                                    | ~15 a 30 min (indexar + benchmark de 30Q)                                                           | ~2h a 4h (montar embeddings + vector store + pipeline)                                                   | **~2h30 a 4h30**           |
| **6** | Guardrails                             | **Não existe**                                    | ~15 a 30 min (benchmark de 30Q antes/depois)                                                        | ~2h a 3h (camadas de bloqueio/reescrita/classificação)                                                   | **~2h30 a 3h30**           |

## Veredito para terminar hoje

- **Q2 + Q3 hoje: totalmente viável.** Somando correções + GPU dá ~3h a 5h no total, e o tempo de GPU em si é curto (o Q1 é que era o gargalo de 5h, não estas). Isso fecha o que o TODO e os PLANOS 1 a 4 definem como escopo ("finalizar completamente as questões 1, 2 e 3"). É o caminho recomendado.
- **Q4 + Q5 + Q6 hoje também: não realista.** São três aplicações do zero, e a Q4 (destilação com dataset sintético + benchmark de 100 perguntas avaliando teacher e student antes/depois) sozinha é meio dia de trabalho. Tentar as seis hoje compromete a qualidade de todas.

Pelo que está nos planos e no TODO, o alvo real de hoje parece ser **Q2 e Q3**. O primeiro bloqueio da Q2 é a fonte de dados errada (`corpus_treino.jsonl` local, que não existe na nuvem) e o campo `topico_inferido` inexistente: sem corrigir isso, o notebook não roda como está.
