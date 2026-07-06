# Trabalho Final — Tópicos em Inteligência Artificial (UFPI, 2026.1)

Adaptação de modelos de linguagem da família **Qwen2.5** ao domínio público
piauiense, cobrindo as seis questões do trabalho final da disciplina de Tópicos em
Inteligência Artificial (Prof. Raimundo Santos Moura).

**Grupo:** Allycia, Heverton, Izaías, Oscar.

## Visão geral das questões

| # | Questão | Técnica | Dataset | Resultado principal |
|---|---------|---------|---------|---------------------|
| 01 | Pré-treino continuado | Treino não supervisionado | DOMPI-2025 | Perplexidade 21,78 → **2,09**; acurácia de tokens 42,25% → **82,03%** |
| 02 | Pós-treino (SFT) | Ajuste fino completo | docentesDC | Perplexidade (resposta) 14,57 → **7,49**; acurácia 51,28% → **63,84%** |
| 03 | Pós-treino (LoRA/QLoRA) | QLoRA (4 bits) | docentesDC | Perplexidade **6,40** com 0,44% dos parâmetros; bônus 1.5B: **4,75** |
| 04 | Destilação de conhecimento | CoT distillation + LoRA | DOMPI-2025 (sintético) | Aluno 5,49 → **4,23** (−22,9%); divergência substring (42%) × juiz (62%) |
| 05 | RAG | Standard RAG (Ragas) | docentesDC | Fidelidade 0,674 / relevância 0,561 / precisão 0,430 (30 perguntas) |
| 06 | Guardrails | Trilhos entrada/tópico/saída | docentesDC | **100%** de detecção de ameaças, 5% de falsos positivos, 95% de disponibilidade |

As Questões 01 a 03 rodaram em uma única sessão de GPU T4 no Kaggle (notebook
unificado, 5h34). As Questões 04 a 06 foram desenvolvidas separadamente pelos
integrantes do grupo, com modelos, ferramentas e ambientes próprios.

## Estrutura do repositório

```
topicos-em-ia/
├── relatorio/            Relatório LaTeX unificado (formato SBC) das Questões 01 a 06
│   ├── main.tex          Documento principal
│   ├── sections/         Seções (01-introducao … 11-consideracoes)
│   ├── figures/          Figuras + gerar_figuras.py (lê os JSON/CSV das questões)
│   └── refs.bib
├── slides/               Apresentação Beamer (tema metropolis) das Questões 01 a 06
│   └── slides.tex
├── questoes/
│   ├── q1-pretreino/     Notebook, resultados (JSON) e figuras da Questão 01
│   ├── q2-sft/           Notebook, resultados e figuras da Questão 02
│   ├── q3-lora-qlora/    Notebook, resultados, figuras e adaptadores da Questão 03 (+ bônus 1.5B)
│   ├── q4-destilacao/    Notebook, dados sintéticos, resultados, adaptador e relatório de origem
│   ├── q5-rag/           Notebook, dados (docentesDC), resultados (CSV) e análise do RAG
│   └── q6-guardrails/    Notebook e resultados dos guardrails
├── docs/                 Enunciado (TRABALHO FINAL.md) e análises de resultados
└── _legado/              Versões antigas e artefatos pesados (fora do git)
```

Cada pasta em `questoes/` é autocontida: `resultados/` guarda as métricas em JSON/CSV
e `figuras/` guarda os gráficos gerados pelo respectivo notebook.

## Como compilar o relatório

```bash
cd relatorio
# opcional: regenerar as figuras a partir dos resultados das questões
python3 figures/gerar_figuras.py
latexmk -pdf main.tex        # gera main.pdf (26 páginas)
```

## Como compilar os slides

```bash
cd slides
latexmk -pdf slides.tex      # gera slides.pdf (requer o tema metropolis e FiraSans)
```

## Observações

- Os pesos completos dos modelos, os `.zip` e os arquivos `.safetensors` não são
  versionados (ver `.gitignore`); os adaptadores LoRA leves ficam em
  `questoes/q3-lora-qlora/adaptadores/` e `questoes/q4-destilacao/adaptador-lora/`.
- Os datasets grandes da Questão 05 (`questoes/q5-rag/dados/`) e a pasta `_legado/`
  são mantidos apenas localmente.
- Todos os números dos relatórios e slides provêm dos artefatos de execução em
  `questoes/*/resultados/`; nenhuma métrica é estimada.
