# Análise completa dos resultados finais (run unificado Q1+Q2+Q3)

> Run de 2026-07-02 no Kaggle (1x T4), notebook unificado, 5h34 de execução total. Artefatos extraídos em `resultados/`. Análise feita por verificação direta dos JSONs, do log completo (`.logs/notebook14006d0a9a.log`) e das 16 figuras, com recomputo independente das métricas derivadas.

## Veredito geral

**Os resultados estão bons e completos. Nada precisa ser re-rodado na GPU.** As três questões têm evidência quantitativa e qualitativa de antes/depois, a lacuna da rodada de junho (benchmark do modelo treinado da Q1) foi fechada, e a única falha do run (LoRA puro da Q3, incompatibilidade de biblioteca do próprio Kaggle) foi absorvida pelo fallback sem comprometer o enunciado, que pede "LoRA e/ou QLoRA". O que resta são melhorias locais de apresentação (5 figuras a regerar a partir dos JSONs, sem GPU) e ressalvas a escrever no relatório.

| Experimento | PPL antes | PPL depois | Acc antes | Acc depois | Tempo | VRAM pico | Params treinados |
|---|---|---|---|---|---|---|---|
| Q1 Pré-treino (full-text) | 21.78 | **2.09** | 42.25% | **82.03%** | 4h23 | 10.6 GB | 494M (100%) |
| Q2 Full SFT (response-only) | 14.57 | **7.49** | 51.28% | **63.84%** | 11min | 9.77 GB | 494M (100%) |
| Q3 QLoRA 0.5B (response-only) | 16.28 | **6.40** | 48.78% | **64.37%** | 18min | 1.77 GB | 2.16M (0.44%) |
| Bônus QLoRA 1.5B (response-only) | 10.28 | **4.75** | 55.82% | **68.44%** | 26min | 2.75 GB | 4.36M (0.28%) |

Atenção: a métrica da Q1 é sobre a sequência completa (modelagem de linguagem) e as de Q2/Q3/bônus são só sobre os tokens de resposta (com máscara). **Não compare PPL da Q1 com as demais no relatório.**

## Questão 1: reproduziu junho e fechou a lacuna

- **Reprodutibilidade quase perfeita:** métricas "antes" idênticas às de junho em 4 casas decimais; "depois" diverge só na 3ª/4ª casa (EC 0.7381 vs 0.7390; acc 82.03% vs 82.00%). Isso é evidência de robustez para citar no relatório.
- **Curva de treino saudável:** loss de 2.05 para 0.64, grad_norm estável (1.7 a 4.2), eval_loss caindo entre épocas (0.8423 na época 1, 0.7356 na época 2). Sem overfitting; o melhor checkpoint é o último.
- **Benchmark de 25 QA agora estruturado nos dois modelos** (a pendência de junho). Classificação item a item (exato / parcial / erro):
  - **Base: 0 / 5 / 20.** Só definições genéricas de blog.
  - **Treinado: 6 / 7 / 12.** Acertos exatos verificados por substring: CNPJ de Aroazes (06.554.984/0001-39), CEP de Aroazes (64.310-000), endereço de Aroazes, CNPJ de Assunção do Piauí (01.612.561/0001-04), CEP de Anísio de Abreu (64.780-000) e a Lei 14.133/2021. Isso supera a expectativa da rodada de junho de que "QA factual ficaria fraco antes e depois": houve memorização factual real do corpus.
- **Limitações a escrever no relatório:**
  - O treinado **alucina com formato plausível**: no item 4 gera um CNPJ bem formatado porém errado para a Câmara de Juazeiro do Piauí, e no item 11 responde "18 territórios" quando são 12. Memorização parcial gera falsa confiança; isso rende um parágrafo de limitação melhor do que a simples ignorância do modelo base.
  - As gerações livres reproduzem **ruído de OCR do corpus** ("REs0LVE", "Licitagao", cedilhas perdidas). Não é degradação do treino, é fidelidade ao DOMPI-2025; explicar para a banca.
  - Nos itens conceituais, o treinado continua o estilo de diário oficial em vez de responder (é um modelo base, sem instruction tuning). Enquadrar o benchmark como evidência qualitativa, não como acurácia de QA.

## Questão 2: sólida, com overfitting já neutralizado

- **Pipeline íntegro:** 13.762 registros brutos, 13.297 após limpeza/tamanho, 11.313 após dedup (2 mil quase-duplicatas removidas), classificados em 7.119 PT / 2.721 inglês / 1.473 código, 3.345 amostrados com cap por professor, **1.977 pares** (A 700, B 500, C 377, D 200, E 200; quase 2x o mínimo de 1.000). 81 descartados por prompt longo: 1.728 treino / 168 avaliação efetivos. **Zero vazamento treino/eval** (verificado programaticamente em três níveis).
- **Quantitativo:** PPL response-only de 14.57 para 7.49 (-48.6%), acurácia de tokens de 51.28% para 63.84%.
- **Overfitting na época 3, já tratado:** eval_loss 0.9528 -> 0.9142 -> 0.9920 enquanto a train loss caiu para ~0.41. O `load_best_model_at_end` restaurou o checkpoint da época 2, então as métricas reportadas já são do modelo sem overfitting. **Re-rodar com 2 épocas produziria o mesmo modelo final** e ainda apagaria a curva de overfitting, que é material de discussão valioso. Não re-rodar.
- **Qualitativo (10 perguntas, greedy): 6 acertos, 2 parciais, 2 erros.** Antes: 0/10 (o base divagava, respondia em inglês ou cuspia código pandas). Depois responde "Raimundo Santos Moura" exato, atribui trechos held-out aos professores corretos (Erico Meneses Leão, Ivan Saraiva Silva) e lista professores por tema.
- **Limitações de qualidade de dados a documentar (não bloqueiam, viram seção de limitações):**
  - Rótulos de **tema (tipo C) são por documento**, não por trecho: um chunk sobre tuplas em Python saiu rotulado "Banco de Dados" (o "erro" do modelo no item 8 da qualitativa é parcialmente artefato do gabarito).
  - **Autoria (tipos A/E) significa "dono do acervo"**, não autor literal (há slide de outra professora dentro da pasta de um docente). Definir a tarefa no relatório como "identificar de qual acervo o material vem".
  - **Tipo D repetitivo:** 200 pares a partir de ~45 fatos únicos (~6x cada), e só 4 pares D no eval.
  - Ruído de OCR nos inputs (mojibake, ligaduras quebradas) e alguns falsos positivos de "código" no tipo E.
  - O eval mede **generalização intra-documento** (chunks inéditos de documentos vistos), que é o objetivo da tarefa, mas não pode ser vendido como generalização para documentos totalmente novos.
- **Nota metodológica obrigatória:** o eval_loss do Trainer (0.91) e a EC final (2.01) usam ponderações/truncamentos diferentes; não plotar juntos sem explicar. Idem para a Q3.

## Questão 3 + bônus: a tese da questão comprovada, com uma ressalva honesta

- **QLoRA 0.5B venceu o Full SFT em perplexidade final (6.40 vs 7.49) treinando 0.44% dos parâmetros com 5.5x menos VRAM** (1.77 vs 9.77 GB) e adapter de 8.7 MB (~113x menor que o modelo). A vitória é genuína e explicável: o Full SFT overfitou na época 3 (eval_loss 0.9142 -> 0.9920) enquanto o QLoRA caiu monotonicamente até 0.8778 (o rank 16 age como regularização num treino de só 1.728 exemplos). Confundidores a citar: LRs diferentes (2e-4 vs 5e-5, escolha padrão de cada técnica) e baselines diferentes (int4 vs fp32).
- **A ressalva honesta: a avaliação qualitativa inverte o ranking.** Placar de acertos nas 10 perguntas: **Full SFT 6/10, QLoRA 0.5B 2/10, QLoRA 1.5B 5/10**. Atualizar todos os pesos memoriza fatos melhor; o adapter aprende o formato e a distribuição, mas retém menos conhecimento factual no 0.5B. O relatório deve apresentar PPL e qualitativa lado a lado; afirmar "QLoRA supera o Full SFT" só pela PPL seria enganoso. O 1.5B QLoRA é o melhor dos dois mundos (PPL 4.75, 5/10 na qualitativa, 2.75 GB).
- **Efeito da quantização isolado:** no mesmo eval set, o base fp32 tem PPL 14.57 e o base NF4 16.28 (+11.7%, -2.5 pp de acurácia). E o 1.5B quantizado (10.28) é melhor que o 0.5B em fp32 (14.57): escala vence precisão numérica.
- **LoRA puro não rodou** (ImportError: o Kaggle tem torchao 0.10.0 pré-instalado e o peft exige 0.16+ no despacho de módulos fp32; no caminho quantizado o dispatcher do bitsandbytes resolve antes). Impacto real: perdeu-se 1 coluna nice-to-have da comparação e a figura de quantização. O enunciado pede "LoRA e/ou QLoRA", então **a questão está cumprida**. Documentar a falha como limitação de infraestrutura (com a linha do log como evidência).
- **Itens 4 e 7 da qualitativa: conferir o gabarito antes de contar como erro.** Os três modelos treinados convergem para a mesma resposta que contradiz a referência, o que sugere referência inconsistente com os pares de treino, não erro dos modelos.

## Figuras: 16 corretas, 5 merecem regeneração local (sem GPU)

Todos os valores anotados batem com os JSONs (recomputados de forma independente, incluindo as somas de 77.337 docs e 13.762 linhas). Problemas são de apresentação:

1. **`q3_metricas.png` (prioridade máxima):** a barra "Base (antes)" é o baseline int4 (16.28), mas o Full SFT partiu do fp32 (14.57). Regerar com 4 barras (Base fp32, Base int4, Full SFT, QLoRA) ou renomear.
2. **`q3_curvas_loss.png` e `q3_painel_comparativo.png`:** títulos prometem "LoRA" que não existe no gráfico. Corrigir título.
3. **`q3_lora_params.png` (vai para o LaTeX):** a escala log faz 228x parecer ~3x; anotar o percentual treinável (100% / 0.44% / 0.28%) ao lado dos valores e rotular o eixo x.
4. **`q1_benchmark_qa.png`:** anotar n por categoria (3/5, 1/5...) e avisar que categorias conceituais zeram por construção da métrica de substring sobre referências longas.
5. **`resumo_geral.png`:** anotar valores nas barras e marcar que Q1 é full-text e as demais response-only.
6. **Criar `q3_quantizacao.png`** (não foi gerada porque dependia do LoRA fp32): 2 barras, fp32 14.57 vs int4 16.28; o dado existe.
7. Cosméticos opcionais: legendas das linhas de corte nos histogramas, rótulos de eixo x nas EDAs.

Também vale **corrigir `resultados_lora_q3.json`**, preenchendo `base.fp32` com os valores medidos antes do crash (EC 2.6793, PPL 14.57, acc 51.28; log linha 1834).

## Decisão sobre re-rodar

| Candidato | Precisa? | Motivo |
|---|---|---|
| Q1 | **Não** | Reproduziu junho dentro do ruído; lacuna do benchmark fechada; curva saudável |
| Q2 (2 épocas) | **Não** | O load_best já entregou o checkpoint da época 2; re-rodar daria o mesmo modelo e apagaria a curva de overfitting, que é discussão útil |
| Q3 LoRA puro | **Não (opcional)** | QLoRA cumpre o enunciado; o baseline fp32 já foi medido; a comparação de quantização existe. Se o relatório ficar pronto cedo e sobrar cota, um run de ~1h30 com `RODAR_Q1=False` e a desinstalação do torchao velho na célula de instalação adiciona a coluna LoRA como apêndice. Não é bloqueante |
| Figuras | **Não (regeração local)** | Todos os dados estão nos JSONs; 15 min de matplotlib local |

## Checklist antes do relatório

- [ ] Regerar as 5 figuras + criar a de quantização (local, a partir dos JSONs)
- [ ] Corrigir `base.fp32` em `resultados_lora_q3.json`
- [ ] Conferir gabarito dos itens 4 e 7 da avaliação qualitativa contra os pares de treino
- [ ] Relatório: tabela do benchmark Q1 (0/5/20 vs 6/7/12) + os 6 acertos factuais exatos
- [ ] Relatório: parágrafo de alucinação plausível (CNPJ errado bem formatado, "18 territórios")
- [ ] Relatório: PPL e qualitativa lado a lado na comparação Full SFT vs QLoRA, com os confundidores (LR, baseline int4)
- [ ] Relatório: seção de limitações da Q2 (rótulo por documento, autoria = acervo, tipo D repetitivo, OCR, generalização intra-documento)
- [ ] Relatório: nota de que Q1 é full-text e Q2/Q3/bônus são response-only (escalas não comparáveis) e de que eval_loss do Trainer difere da EC final por ponderação
- [ ] Relatório: registrar a falha do LoRA puro (torchao 0.10.0 do Kaggle) como limitação de infraestrutura
- [ ] Não afirmar warmup de 5% no texto: na versão do transformers do Kaggle o warmup efetivo foi de 0 passos (sem impacto visível nas curvas, mas o texto precisa ser fiel)
