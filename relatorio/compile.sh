#!/usr/bin/env bash
# Compila o documento e reporta status.
# Se existe document/main.tex (layout export-ready para Overleaf),
# compila dentro de document/ e copia main.pdf para a raiz.
# Caso contrário, compila main.tex da raiz (layout simples).
#
# Uso: ./compile.sh [--quiet]

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

if [[ -f document/main.tex ]]; then
  SRC_DIR="document"
elif [[ -f main.tex ]]; then
  SRC_DIR="."
else
  echo "❌ nem document/main.tex nem main.tex existem. Rode ./new-doc.sh <template> primeiro." >&2
  exit 1
fi

QUIET=0
[[ "${1:-}" == "--quiet" ]] && QUIET=1

if ! command -v latexmk >/dev/null; then
  echo "❌ latexmk não encontrado. Instale TeX Live (texlive-full)." >&2
  exit 1
fi

LOG=$(mktemp)
trap 'rm -f "$LOG"' EXIT

compile() {
  (cd "$SRC_DIR" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex "$@")
}

publish_pdf() {
  if [[ "$SRC_DIR" != "." && -f "$SRC_DIR/main.pdf" ]]; then
    cp "$SRC_DIR/main.pdf" "$PROJECT_ROOT/main.pdf"
  fi
}

if [[ $QUIET -eq 1 ]]; then
  if compile >"$LOG" 2>&1; then
    publish_pdf
    PAGES=$(pdfinfo main.pdf 2>/dev/null | awk '/^Pages:/ {print $2}')
    echo "✅ compila (${PAGES:-?} páginas)"
  else
    ERR=$(grep -E "^!" "$LOG" | head -1)
    echo "❌ falha: ${ERR:-erro desconhecido (ver $SRC_DIR/main.log)}"
    exit 1
  fi
else
  compile
  publish_pdf
  PAGES=$(pdfinfo main.pdf 2>/dev/null | awk '/^Pages:/ {print $2}')
  echo ""
  echo "📄 PDF gerado: main.pdf (${PAGES:-?} páginas)"
fi
