/******************************************************************************
*  FFT Tuning Command Version 6.5
*
*  Copyright (c) 2015 OK Ojisan(Takuya OKAHISA)
*  Released under the MIT license
*  http://opensource.org/licenses/mit-license.php
******************************************************************************/

#include <iostream>
#include <iomanip>
#include <fstream>

#ifdef __AVX__
#define N_MAX 21
#define FACTOR 4
#else
#define N_MAX 22
#define FACTOR 3
#endif

int main(){
    using namespace std;
    static const int n_min  = 1;
    static const int n_max  = N_MAX;

    ofstream fs1("otfft_setup.h");
    ofstream fs2("otfft_fwd.h");
    ofstream fs3("otfft_inv.h");
    ofstream fs4("otfft_fwd0.h");
    ofstream fs5("otfft_invn.h");
    fs1 << "switch (log_N) {\ncase  0: break;\n";
    fs2 << "switch (log_N) {\ncase  0: break;\n";
    fs3 << "switch (log_N) {\ncase  0: break;\n";
    fs4 << "switch (log_N) {\ncase  0: break;\n";
    fs5 << "switch (log_N) {\ncase  0: break;\n";

    // hankel single thread 1-22
    int optimal_engines[]={
      0, 6, 7, 7, 2, 4, 3, 7,
      2, 3, 1, 6, 4, 4, 4, 4,
      4, 4, 4, 6, 4, 4, 4, };

    for (int n = n_min; n <= n_max; n++) {
        int fft_num = optimal_engines[n];
        fs1 << "case " << setw(2) << n << ": fft" << fft_num << "->setup2(log_N); break;\n";
        fs2 << "case " << setw(2) << n << ": fft" << fft_num << "->fwd(x, y); break;\n";
        fs3 << "case " << setw(2) << n << ": fft" << fft_num << "->inv(x, y); break;\n";
        fs4 << "case " << setw(2) << n << ": fft" << fft_num << "->fwd0(x, y); break;\n";
        fs5 << "case " << setw(2) << n << ": fft" << fft_num << "->invn(x, y); break;\n";
        cout << " fft" << fft_num << endl;
    }

#ifndef DO_SINGLE_THREAD
    fs1 << "default: fft5->setup2(log_N); break;\n";
    fs2 << "default: fft5->fwd(x, y); break;\n";
    fs3 << "default: fft5->inv(x, y); break;\n";
    fs4 << "default: fft5->fwd0(x, y); break;\n";
    fs5 << "default: fft5->invn(x, y); break;\n";
#else
    fs1 << "default: fft4->setup2(log_N); break;\n";
    fs2 << "default: fft4->fwd(x, y); break;\n";
    fs3 << "default: fft4->inv(x, y); break;\n";
    fs4 << "default: fft4->fwd0(x, y); break;\n";
    fs5 << "default: fft4->invn(x, y); break;\n";
#endif
    fs1 << "}\n";
    fs2 << "}\n";
    fs3 << "}\n";
    fs4 << "}\n";
    fs5 << "}\n";

    return 0;
}
