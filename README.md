#OTFFT
[Stockham FFT アルゴリズムの解説](http://www.moon.sannet.ne.jp/okahisa/stockham/stockham.html) で紹介されている「OTFFT(OKおじさん Template FFT ライブラリ)」を使いやすいように弄ったものです。

##ライセンス
the MIT License

##使い方

例:
```bash
PREFIX=/usr/local
make all
sudo make install INSDIR=$PREFIX
g++ -march=native -I $PREFIX/include -c prog.cpp
g++ -march=native -o prog prog.o -L $PREFIX/lib -lotfft -Wl,-R,$PREFIX/lib
```

----
以下はオリジナル OTFFT の README です。

This software is released under the MIT License, see LICENSE.txt.

このソフトウェアは MIT ライセンスのもとにリリースされています。詳細は LICENSE.txt を見てください。


##使用上の注意

　OTFFT は、コンパイルする環境に最適化されます。つまり、AVX が有効な環境でコンパイルすれば、AVX を使うようになります。このバイナリを AVX をサポートしない環境で動かせば、当然、例外を起こして落ちます。そのため、コンパイルしたバイナリを、広く一般に配布するような利用形態には、OTFFT は向きません。OTFFTは、数値計算のプログラムをソースからコンパイルするような利用形態を想定しています。


##必要なファイル

　`otfft-6.5.tar.gz` を展開すると、いくつかのファイルができますが、OTFFT を使うのに必要なファイルはその中の `otfft` フォルダとその中身です。このうち、

    otfft_setup.h
    otfft_fwd.h
    otfft_fwd0.h
    otfft_inv.h
    otfft_invn.h

は自動生成されるファイルで、自分の環境で最大の効果を発揮したければ`ffttune` を使って生成し直してやる必要があります。まず、otfft フォルダで

    make ffttune

を実行し、`ffttune` コマンドを作ります。`Makefile` は自分の環境に合わせていじってください。C++ の template によるメタプログラミング技法を用いているので、コンパイルには非常に長い時間がかかります。次に

    ./ffttune

と ffttune コマンドを実行します。すると先に示したファイルの自動生成が始まります。しばらくお待ちください。このとき、Core i7(4 コア 8 スレッド) の環境の場合、環境変数 OMP_NUM_THREADS を 7 に設定する必要があるかもしれません。一部の Linux では、8 スレッドで動かすと謎の速度低下を起こします。

　OTFFT は内部で幾つかのアルゴリズムを使い分けていますが、どのアルゴリズムがどのサイズで一番速いか、実際に計測して決定します。しかし、パソコンは FFT だけを黙々と実行しているわけではなく、裏で色々なプロセスが動いています。計測には常に突発的なゆれが混入します。そのため、１回の計測で最適な組み合わせが求まるとは限りません。納得いかない場合は何度か `ffttune` を実行し直してください。

　自動生成が終わったら、次は `otfft.o` ファイルを生成します。`otfft` フォルダで次のコマンドを実行します。

    make otfft.o

　C++ の template によるメタプログラミング技法を用いているので、コンパイルには非常に長い時間がかかります。辛抱強くお待ちください。

　これで準備は終わりです。バージョン 3.2 までは OTFFT は全てヘッダファイルで実装されていたため、ご自分のバイナリで利用する場合でも、いちいちコンパイルし直されていました。そのため、上記のように非常に長いコンパイル時間が必要でした。バージョン 4.0 から OTFFT 自身のコンパイルは事前に済ませておくようにしました。ですから、ご自分のバイナリで利用する場合はリンクするだけです。


##シングルスレッドモード

　OTFFT はデフォルトでは OpenMP によるマルチスレッドで動作します。しかし、
中にはマルチスレッドの実装は、自前の仕組みを使いたい人もいるかもしれません。
そこで、シングルスレッドで動作するモードも用意しました。OTFFT は
`DO_SINGLE_THREAD` マクロを定義してコンパイルすると、シングルスレッドモードに
なります。シングルスレッドモードにしたい場合は、`otfft/otfft_misc.h` で定義して、
`ffttune` コマンドの生成から始めてください。


##ご自分のバイナリのコンパイル方法

　さて、`otfft.o` の生成が終われば、あとは `otfft/otfft.h` をインクルードし、コンパイラのインクルードパスに otfft フォルダのあるフォルダを加えて、`otfft.o` ファイルを目的のバイナリにリンクしてやれば OTFFT は使えます。

　g++ でコンパイルするには、-lgomp や -liomp5 など OpenMP のランタイムもリンクする必要があります。例えば、カレントフォルダに otfft フォルダがあるなら以下のようにコンパイルします。

```bash
g++ -c -Ofast hello.cpp
g++ hello.o otfft/otfft.o -lgomp -o hello
```


##複素離散フーリエ変換

　サイズ `N` の複素離散フーリエ変換を実行するには、

```cpp
#include "otfft/otfft.h"
using OTFFT::complex_t;
using OTFFT::simd_malloc;
using OTFFT::simd_free;

void f(int N)
{
    complex_t* x = (complex_t*) simd_malloc(N*sizeof(complex_t));
    // 何かする...
    OTFFT::FFT fft(N); // FFT オブジェクトの作成
    fft.fwd(x);        // 複素離散フーリエ変換を実行。x が入力かつ出力
    // 何かする...
    simd_free(x);
}
```

のようにします。`N` は最大 2^24 (2の24乗)までで、2のべき乗である必要があります。

`complex_t` は

```cpp
struct complex_t
{
    double Re, Im;

    complex_t() : Re(0), Im(0) {}
    complex_t(const double& x) : Re(x), Im(0) {}
    complex_t(const double& x, const double& y) : Re(x), Im(y) {}
    complex_t(const std::complex<double>& z) : Re(z.real()), Im(z.imag()) {}
    operator std::complex<double>() { return std::complex<double>(Re, Im); }

    // その他のメンバ関数...
};
```

のように定義されています。

　フーリエ変換(`fwd`)には係数 `1/N` が掛かっています。もしそれが不都合なら係数の掛かっていない(`fwd0`)もあります。以下のようなメンバ関数が用意されています。

    fwd0(x) 離散フーリエ変換(正規化無し)
    fwd(x)  離散フーリエ変換(1/N による正規化付き)
    fwdn(x) 離散フーリエ変換(1/N による正規化付き)

    inv0(x) 逆離散フーリエ変換(正規化無し)
    inv(x)  逆離散フーリエ変換(正規化無し)
    invn(x) 逆離散フーリエ変換(1/N による正規化付き)

FFT のサイズを変更したい場合は、

    fft.setup(2 * N);

のようにします。

　OTFFT は Stockham のアルゴリズムを使っていますので、実行には入力系列と同じサイズの作業領域が必要です。普通は内部でその領域を確保しますが、マルチスレッドのプログラムを組む場合など、それだと都合が悪い場合があります。そんな時は外部から作業領域を渡すバージョンもあります。次のように使います。

```cpp
#include "otfft/otfft.h"
using OTFFT::complex_t;
using OTFFT::simd_malloc;
using OTFFT::simd_free;

void f(int N)
{
    complex_t* x = (complex_t*) simd_malloc(N*sizeof(complex_t));
    complex_t* y = (complex_t*) simd_malloc(N*sizeof(complex_t));
    // 何かする...
    OTFFT::FFT0 fft(N);
    fft.fwd(x, y); // x が入力、y が作業領域
    // 何かする...
    fft.inv(x, y); // x が入力、y が作業領域
    // 何かする...
    simd_free(y);
    simd_free(x);
}
```

　`OTFFT::FFT` だったところが `OTFFT::FFT0` になっていることに注意してください。


##実離散フーリエ変換

　バージョン 3.0 から実離散フーリエ変換、離散コサイン変換(DCT-II)、Bluestein's FFT(任意サイズの FFT) もサポートするようになりました。サイズ `N` の実離散フーリエ変換を実行するには以下のようにします。

```cpp
#include "otfft/otfft.h"
using OTFFT::complex_t;
using OTFFT::simd_malloc;
using OTFFT::simd_free;

void f(int N)
{
    double*    x = (double*)    simd_malloc(N*sizeof(double));
    complex_t* y = (complex_t*) simd_malloc(N*sizeof(complex_t));
    // 何かする...
    OTFFT::RFFT rfft(N);
    rfft.fwd(x, y); // 実離散フーリエ変換を実行。x が入力、y が出力
    // 何かする...
    simd_free(y);
    simd_free(x);
}
```

　実離散フーリエ変換には以下のようなメンバ関数が用意されています。

    fwd0(x, y) 実離散フーリエ変換(正規化無し)           x:入力、y:出力
    fwd(x, y)  実離散フーリエ変換(1/N による正規化付き) x:入力、y:出力
    fwdn(x, y) 実離散フーリエ変換(1/N による正規化付き) x:入力、y:出力

    inv0(y, x) 逆実離散フーリエ変換(正規化無し)           y:入力、x:出力
    inv(y, x)  逆実離散フーリエ変換(正規化無し)           y:入力、x:出力
    invn(y, x) 逆実離散フーリエ変換(1/N による正規化付き) y:入力、x:出力

　逆実離散フーリエ変換は複素系列 `y` を受け取って、実系列 `x` を返します。`y` が斜対称、すなわち `y[N-k] == conj(y[k])` でないと正しい結果を返しません。また、逆実離散フーリエ変換は入力 `y` を破壊します。保存しておきたい場合は、コピーを取っておく必要があります。指定できるサイズは２のべき乗で2^25 以下です。内部的には `N/2` のサイズの複素 FFT で実装されています。

　実離散フーリエ変換は、作業領域として出力系列を使います。そして逆実離散フーリエ変換は、作業領域として入力系列を使います。マルチスレッドでプログラムする場合も、それぞれのスレッド専用の入力と出力を与えてやれば OK です。


##離散コサイン変換(DCT-II)

　サイズ `N` の離散コサイン変換(DCT-II)を実行するには以下のようにします。

```cpp
#include "otfft/otfft.h"
using OTFFT::complex_t;
using OTFFT::simd_malloc;
using OTFFT::simd_free;

void f(int N)
{
    double* x = (double*) simd_malloc(N*sizeof(double));
    // 何かする...
    OTFFT::DCT dct(N);
    dct.fwd(x); // DCT-II を実行する。x が入力かつ出力
    // 何かする...
    simd_free(x);
}
```

　離散コサイン変換には以下のようなメンバ関数が用意されています。

    fwd0(x) 離散コサイン変換(正規化無し)
    fwd(x)  離散コサイン変換(1/N による正規化付き)
    fwdn(x) 離散コサイン変換(1/N による正規化付き)

    inv0(x) 逆離散コサイン変換(正規化無し)
    inv(x)  逆離散コサイン変換(正規化無し)
    invn(x) 逆離散コサイン変換(1/N による正規化付き)

　離散コサイン変換は DCT-II を採用しています。ただし、直交化はしていません。指定できるサイズは２のべき乗で 2^25 以下です。内部的には `N/2` のサイズの複素FFT で実装されています。マルチスレッドで使う場合の作業領域を外部から与えるバージョンもあります。以下のように使います。

```cpp
#include "otfft/otfft.h"
using OTFFT::complex_t;
using OTFFT::simd_malloc;
using OTFFT::simd_free;

void f(int N)
{
    double*    x = (double*)    simd_malloc(N*sizeof(double));
    double*    y = (double*)    simd_malloc(N*sizeof(double));
    complex_t* z = (complex_t*) simd_malloc(N*sizeof(complex_t));
    // 何かする...
    OTFFT::DCT0 dct(N);
    dct.fwd(x, y, z); // DCT-II を実行する。x が入力かつ出力、y,z が作業領域
    // 何かする...
    simd_free(z);
    simd_free(y);
    simd_free(x);
}
```

　`OTFFT::DCT` だったところが `OTFFT::DCT0` になっていることに注意してください。


##Bluestein's FFT

　Bluestein's FFT(任意サイズ `N` の FFT) を実行するには以下のようにします。

```cpp
#include "otfft/otfft.h"
using OTFFT::complex_t;
using OTFFT::simd_malloc;
using OTFFT::simd_free;

void f(int N)
{
    complex_t* x = (complex_t*) simd_malloc(N*sizeof(complex_t));
    // 何かする...
    OTFFT::Bluestein bst(N); // N は任意の自然数。ただし、2^23 より小さい。
    bst.fwd(x); // Bluestein's FFT を実行する。x が入力かつ出力
    // 何かする...
    simd_free(x);
}
```

　Bluestein's FFT では離散フーリエ変換のサイズを２のべき乗に限る必要はありません。例えば大きな素数のサイズでも計算量は O(N log N) になります。指定出来るサイズの上限は 2^23 です。だたし、マルチスレッド用のメンバ関数は用意されていません。どうしてもマルチスレッドで使いたい場合は、動作確認はしていませんが、スレッドの数だけ Bluestein オブジェクトを生成し、各スレッドでそれらを使い分ければ大丈夫と思います。当然、メモリをゴージャスに使ってしまいます。

　Bluestein's FFT には以下のようなメンバ関数が用意されています。

    fwd0(x) 離散フーリエ変換(正規化無し)
    fwd(x)  離散フーリエ変換(1/N による正規化付き)
    fwdn(x) 離散フーリエ変換(1/N による正規化付き)

    inv0(x) 逆離散フーリエ変換(正規化無し)
    inv(x)  逆離散フーリエ変換(正規化無し)
    invn(x) 逆離散フーリエ変換(1/N による正規化付き)


##ベンチマーク

　最後に、OTFFT にはベンチマークプログラムが付属しています。しかし、このベンチマークには FFTW3 と大浦さんのプログラムを用いているため、単独では動作しません。まずは FFTW3 をインストールし、大浦さんのページ

    http://www.kurims.kyoto-u.ac.jp/~ooura/fft-j.html

から大浦さんの FFT ライブラリをダウンロードしてください。そしてその中のfftsg.c というファイルを `fftbench1.cpp` と同じフォルダに入れてください。その上で以下のように `fftbench1` コマンドと `fftbench2` コマンドを `make` します。

    make fftbench1 fftbench2

`fftbench1` が計測時間に FFT オブジェクトの初期化を含まないベンチマーク、`fftbench2` が初期化を含むベンチマークです。以下のように実行してください。

    ./fftbench1
    ./fftbench2
