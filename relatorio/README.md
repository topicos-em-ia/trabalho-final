# Relatório - Tópicos em IA (Questões 01, 02 e 03)

Relatório em LaTeX (formato ABNT) sobre pré-treino e pós-treino do `Qwen2.5-0.5B`
no domínio público piauiense. Baseado no fluxo do projeto `/home/izaias/latex`.

## Compilar

```bash
./compile.sh        # gera main.pdf
make build          # idem
make watch          # recompila a cada save
```

Requer `texlive-full` (latexmk, pdflatex, bibtex) e o pacote `abntex2`.

## Estrutura

```
relatorio/
├── main.tex                 # documento principal (preâmbulo + \input das seções)
├── refs.bib                 # bibliografia (abntex2cite, autor-data)
├── compile.sh / Makefile    # build (latexmk -> pdflatex + bibtex)
├── figures/
│   ├── gerar_figuras.py      # gera os gráficos com matplotlib (make figuras)
│   ├── q1_metricas.png
│   └── q3_lora_params.png
└── sections/
    ├── 00-capa.tex
    ├── 01-introducao.tex
    ├── 02-fundamentacao.tex
    ├── 03-metodologia.tex
    ├── 04-questao01.tex      # Q1 (pré-treino) — completa
    ├── 05-questao02.tex      # Q2 (SFT) — baseline + em andamento
    ├── 06-questao03.tex      # Q3 (LoRA/QLoRA) — baseline + em andamento
    ├── 07-discussao.tex
    └── 08-consideracoes.tex
```

## Regenerar gráficos

```bash
make figuras        # ou: cd figures && python3 gerar_figuras.py
```
