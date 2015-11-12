#!/bin/bash

function dispatch.original-web {
  wget -r http://www.moon.sannet.ne.jp/okahisa/stockham/stockham.html

  stockham=www.moon.sannet.ne.jp/okahisa/stockham
  (
    cd "$stockham"
    for f in *.html; do
      iconv -c -f EUC-JP -t UTF-8 "$f" \
        | sed -r 's/\bcharset=EUC-JP\b/charset=utf-8/ ; /^(<IMG SRC="http:\/\/www\.sannet\.ne\.jp\/counter\/c[[:digit:]]\.gif" ALT="[[:digit:]]">)+?$/d' > "$f.part" \
        && touch -r "$f" "$f.part" \
        && mv -f "$f.part" "$f"
    done
  )
  mv -f "$stockham"/* html/
  rm -rf www.moon.sannet.ne.jp
}

function dispatch.modify-source {
  local src="$1"
  if [[ ! -e $src/README.txt ]]; then
    echo "$prog: the specified directory, $src, does not seem to be an original otfft directory." >&2
    return 1
  fi

  #sed -i 's/^[[:space:]]*#pragma[[:space:]]\{1,\}omp[[:space:]]\{1,\}.*$/#ifdef _OPENMP\n&\n#endif/g' "$src"/*.h "$src"/*.cpp "$src"/otfft/*.h
  refact '^[[:space:]]*#pragma[[:space:]]+omp[[:space:]]+.*$' '#ifdef _OPENMP\n&\n#endif' "$src"/*.h "$src"/*.cpp "$src"/otfft/*.h
  refact '\b(otfft|otfft_misc|stopwatch)\.h\b' 'otfft/&' "$src"/otfft/*.h "$src"/otfft/*.cpp
}

function dispatch.help {
  echo "usage: $prog help|original-web|modify-source"
}

prog="${0##*/}"

if (($#==0)); then
  dispatch.help
  exit 1
fi

if declare -f "dispatch.$1" &>/dev/null; then
  "dispatch.$1" "${@:2}"
else
  echo "$prog: unknown subcommand $1" >&2
  dispatch.help >&2
  exit 1
fi
