# Análise Rápida do Run

Deu certo. O run completou de ponta a ponta em 5h34 e todas as três questões têm resultados válidos.

A única falha foi o LoRA puro da Q3, e o fallback absorveu exatamente como projetado: o QLoRA completou e cumpre a questão (o enunciado pede "LoRA e/ou QLoRA").

## Resultados quantitativos

Tabela de consolidação do log:

| Experimento              | PPL antes | PPL depois | Acc antes | Acc depois |
| ------------------------ | --------- | ---------- | --------- | ---------- |
| Q1 Pré-treino (DOMPI)    | 21.78     | 2.09       | 42.25%    | 82.03%     |
| Q2 Full SFT (docentesDC) | 14.57     | 7.49       | 51.28%    | 63.84%     |
| Q3 QLoRA 0.5B            | 16.28     | 6.40       | 48.78%    | 64.37%     |
| Bônus QLoRA 1.5B         | 10.28     | 4.75       | 55.82%    | 68.44%     |

## Q1: reproduziu a rodada anterior e fechou a lacuna

Treino de 4h23 (loss final 0.8943 vs 0.8945 da rodada passada, praticamente idêntico), eval_loss caindo da época 1 (0.8423) para a 2 (0.7356) sem overfitting, melhor checkpoint carregado.

E o dado que faltava agora existe: o benchmark de 25 perguntas foi respondido pelo modelo treinado e salvo em `resultados_benchmark_treinado.json`. Melhor ainda, o resultado superou a expectativa de que "QA factual ficaria fraco": o modelo treinado acertou exatamente o CNPJ da Prefeitura de Aroazes (06.554.984/0001-39), o CNPJ de Assunção do Piauí (01.612.561/0001-04), o CEP 64.310-000 e o endereço Av. 27 de Fevereiro, 691, Centro. O modelo base só divagava. Isso rende um parágrafo forte no relatório.

## Q2: o antes/depois qualitativo ficou dramático

Pipeline gerou os 1.977 pares (1.728 treino / 168 avaliação após filtro). Amostras do log:

- "Quem é o professor com mais material?" Antes: resposta circular sem conteúdo. Depois: "Raimundo Santos Moura." (exato)
- Atribuição de trechos held-out: antes respondia em inglês ou cuspia código; depois: "Este material pertence ao professor Erico Meneses Leao" e "Ivan Saraiva Silva" (ambos corretos, em trechos nunca vistos no treino)
- Tema de trecho do André Soares: "Redes de Computadores" (correto)
- Um erro honesto para citar no relatório: no item 8 respondeu "Engenharia de Software" quando a referência era "Banco de Dados"

Detalhe metodológico valioso: a eval_loss subiu na época 3 (0.914 para 0.992), ou seja, o Full SFT começou a decorar, e o `load_best_model_at_end` recuperou a época 2. O QLoRA, ao contrário, melhorou até o fim. Isso justifica a escolha dos mecanismos e vira análise no relatório.

## Q3: QLoRA venceu o Full SFT gastando 5,5x menos VRAM

O LoRA puro falhou com `ImportError: torchao 0.10.0` (o Kaggle tem um torchao velho pré-instalado e o peft novo o rejeita ao despachar módulos em modelos fp32; no caminho quantizado o dispatcher do bitsandbytes resolve antes, por isso o QLoRA passou). A blindagem imprimiu o traceback e seguiu. O QLoRA entregou:

- PPL final 6.40 contra 7.49 do Full SFT no mesmo conjunto de avaliação, treinando só 2,2M de parâmetros contra 494M
- Pico de VRAM de 1.77 GB contra 9.77 GB do Full SFT, adapter de 8.7 MB contra ~1 GB de modelo
- É literalmente a tese da questão ("LoRA/QLoRA chega perto ou melhor com fração do custo") comprovada nos seus dados

O bônus 1.5B também completou (26 min, 2.75 GB de VRAM) e cumpre o "mais de um modelo com parâmetros diferentes".

## Perdas e saúde do run

- Única figura ausente: `q3_quantizacao.png` (dependia do LoRA fp32). Mas os números da comparação estão no log e nos JSONs: quantizar para NF4 custou +1.71 de PPL no modelo base (14.57 fp32 vs 16.28 int4). Dá para citar no texto mesmo sem o gráfico.
- Infra impecável: VRAM zerada entre seções (0.02 GB), disco nunca abaixo de 18.9 GB livres, 16 figuras + 10 JSONs + zip final gerados, e todos os JSONs pequenos impressos no log como backup. Gastou só 5.6h das 30h de cota.

## Próximos passos

Se você quiser o LoRA puro para completar a comparação de três técnicas, a correção é uma linha (desinstalar o torchao velho na célula de instalação) e um re-run com `RODAR_Q1 = False` custa ~1h30 de GPU. Opcional: a Q3 já está cumprida com o QLoRA.

Quando terminar de baixar os resultados, posso descompactar no repo e atualizar o relatório LaTeX com os números e figuras novos.
