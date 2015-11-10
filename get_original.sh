#!/bin/bash

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
