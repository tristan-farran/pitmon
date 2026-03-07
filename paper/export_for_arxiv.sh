#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$SCRIPT_DIR"
ROOT_DIR="$(cd "$PAPER_DIR/.." && pwd)"
EXPERIMENT_OUT_DIR="$ROOT_DIR/experiment/core/out"
OUT_DIR="${1:-$PAPER_DIR/arxiv}"

SRC_MAIN="$PAPER_DIR/main.tex"
DST_MAIN="$OUT_DIR/main.tex"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/assets"

cp "$PAPER_DIR/references.bib" "$OUT_DIR/"

# Convert EJS-oriented source to a standalone article for ArXiv.
perl -0777 -pe '
  s/\\documentclass\[ejsv2\]\{imsart\}/\\documentclass[11pt]{article}/g;
  s/^\\runtitle\{.*?\}\n//mg;
  s/\\begin\{aug\}.*?\\end\{aug\}/\\author{Tristan Farran\\\\University of Amsterdam}\n\\date{}/s;
  s/^\\address\[.*?\]\{.*?\}\n//mg;
  s/^\\runauthor\{.*?\}\n//mg;
  s/\\begin\{frontmatter\}\n?//g;
  s/\\end\{frontmatter\}\n?//g;
  s/\n\\begin\{abstract\}/\n\\maketitle\n\n\\begin{abstract}/g;
  s/\n\\begin\{keyword\}\[class=MSC\].*?\\end\{keyword\}\n?/\n/s;
  s/\n\\begin\{keyword\}.*?\\end\{keyword\}\n?/\n/s;
  s/^\\startlocaldefs\n?//mg;
  s/^\\endlocaldefs\n?//mg;
  s/\\graphicspath\{\{\.\.\/experiment\/core\/out\/\}\}/\\graphicspath{{assets\/}}/g;
  s/\\input\{\.\.\/experiment\/core\/out\/([^}]+)\}/\\input{assets\/$1}/g;
  s/\\bibliographystyle\{imsart-nameyear\}/\\bibliographystyle{plainnat}/g;
' "$SRC_MAIN" > "$DST_MAIN"

copy_input_asset() {
  local input_ref="$1"
  [[ "$input_ref" == ../experiment/core/out/* ]] || return 0

  local rel="${input_ref#../experiment/core/out/}"
  local src="$EXPERIMENT_OUT_DIR/$rel"
  [[ "$src" == *.tex ]] || src="${src}.tex"
  cp "$src" "$OUT_DIR/assets/$(basename "$src")"
}

copy_graphic_asset() {
  local ref="$1"
  local src

  if [[ "$ref" == ../experiment/core/out/* ]]; then
    src="$ROOT_DIR/${ref#../}"
  else
    src="$EXPERIMENT_OUT_DIR/$ref"
  fi
  cp "$src" "$OUT_DIR/assets/$(basename "$src")"
}

# Copy \input assets
grep -oE '\\input\{[^}]+\}' "$SRC_MAIN" \
  | sed -E 's/.*\{([^}]+)\}.*/\1/' \
  | sort -u \
  | while IFS= read -r r; do
      [[ -n "$r" ]] && copy_input_asset "$r"
    done

# Copy \includegraphics assets
grep -oE '\\includegraphics(\[[^]]*\])?\{[^}]+\}' "$SRC_MAIN" \
  | sed -E 's/.*\{([^}]+)\}.*/\1/' \
  | sort -u \
  | while IFS= read -r g; do
      [[ -n "$g" ]] && copy_graphic_asset "$g"
    done

(
  cd "$OUT_DIR"
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
)

echo "ArXiv package ready at: $OUT_DIR"