.PHONY: help build pdf watch clean

help:
	@echo "Targets disponíveis:"
	@echo "  make build   — compila main.pdf (one-shot, via latexmk)"
	@echo "  make pdf     — alias de build"
	@echo "  make watch   — recompila a cada save (latexmk -pvc)"
	@echo "  make figuras — regenera os gráficos (matplotlib)"
	@echo "  make clean   — remove arquivos auxiliares (preserva main.pdf)"

build pdf:
	./compile.sh

watch:
	latexmk -pdf -pvc -interaction=nonstopmode -halt-on-error main.tex

figuras:
	cd figures && python3 gerar_figuras.py

clean:
	latexmk -c
	rm -f *.bbl *.blg
