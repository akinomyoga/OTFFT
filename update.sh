#!/bin/bash

function mkd { [[ -d $1 ]] || mkdir -p "$1"; }

function dispatch.original-web/remove-counter {
  local dir="$1"
  (
    cd "$dir"
    for f in *.html; do
      [[ -s $f ]] || continue
      # 2016-05-21 EUC-JP -> UTF-8 に変わった様だ。
      cat "$f" \
        | sed -r 's/\bcharset=EUC-JP\b/charset=utf-8/ ; /^(<IMG SRC="http:\/\/www\.sannet\.ne\.jp\/counter\/c[[:digit:]]\.gif" ALT="[[:digit:]]">)+?$/d' > "$f.part" \
        && touch -r "$f" "$f.part" \
        && mv -f "$f.part" "$f"
    done
  )
}

function dispatch.original-web {
  wget -r http://wwwa.pikara.ne.jp/okojisan/stockham/index.html
  stockham=wwwa.pikara.ne.jp/okojisan/stockham
  dispatch.original-web/remove-counter "$stockham"
  mkd html/ja && mv -f "$stockham"/* html/ja

  wget -r http://wwwa.pikara.ne.jp/okojisan/otfft-en/stockham1.html
  otfften=wwwa.pikara.ne.jp/okojisan/otfft-en
  dispatch.original-web/remove-counter "$otfften"
  mkd html/en && mv -f "$otfften"/* html/en

  rm -rf wwwa.pikara.ne.jp
}

function dispatch.modify-source {
  local src="$1"
  if [[ ! -e $src/README.txt ]]; then
    echo "$prog: the specified directory, $src, does not seem to be an original otfft directory." >&2
    return 1
  fi
  if [[ -d $src/src ]]; then
    echo "$prog: source has been already modified." >&2
    return 1
  fi

  #sed -i 's/^[[:space:]]*#pragma[[:space:]]\{1,\}omp[[:space:]]\{1,\}.*$/#ifdef _OPENMP\n&\n#endif/g' "$src"/*.h "$src"/*.cpp "$src"/otfft/*.h

  mv "$src"/otfft "$src"/src
  mkdir -p "$src"/src/otfft
  mv "$src"/src/{stopwatch,otfft,otfft_misc,msleep}.h "$src"/src/otfft
  patch -lcf "$src"/src/otfft/otfft_misc.h < update/otfft_misc.patch
  mkdir -p "$src"/check
  mv "$src"/{{fft,rfft,bst,dct}check.cpp,fftbench{1,2}.cpp,simple_fft.h,cpp_fftw3.h,ooura{1,2}.h} "$src"/check

  # #ifndef USE_FFT_THREADS
  # patch -lcf "$src"/check/fftbench1.cpp < update/fftbench1.cpp.patch
  # patch -lcf "$src"/check/fftbench2.cpp < update/fftbench2.cpp.patch
  # patch -lcf "$src"/check/cpp_fftw3.h < update/cpp_fftw3.h.patch

  mkdir -p "$src"/out/include
  mv "$src"/src/otfft_{fwd,fwd0,inv,invn,setup}.h "$src"/out/include
  refact '^[[:space:]]*#pragma[[:space:]]+omp[[:space:]]+.*$' '#ifdef _OPENMP\n&\n#endif' "$src"/{src,check}/*.{h,cpp} "$src"/src/otfft/*.h
  refact '[[:space:]]+$' '' "$src"/{src,check}/*.{h,cpp} "$src"/src/otfft/*.h
  refact '"\b(otfft|otfft_misc|msleep|stopwatch)\.h\b' '"otfft/\1.h' "$src"/{src,check}/*.{h,cpp}
}

function dispatch.diff {
  local src=${1:-otfft-5.4.20151111}
  local dst=${2:-otfft-6.0.20151226}
  src=${src%/} dst=${dst%/}
  find $src > $src.lst
  find $dst > $dst.lst
  sed -i s,[^/]*/,, $src.lst $dst.lst
  diff -bwu $src.lst $dst.lst
}

function dispatch.help {
  local commands=($(declare -f | sed -n 's/^dispatch\.\([^[:space:]]*\) ()[[:space:]]*/\1/p'))
  IFS="|" eval 'commands="${commands[*]}"'
  echo "usage: $prog $commands"
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
