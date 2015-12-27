/******************************************************************************
*  Copyright (c) 2015 OK Ojisan(Takuya OKAHISA)
*  Released under the MIT license
*  http://opensource.org/licenses/mit-license.php
******************************************************************************/

#ifndef stopwatch_h
#define stopwatch_h

#if defined(_MSC_VER) || defined(__WINNT__)

#include <windows.h>

typedef LONGLONG counter_t;

inline counter_t get_counter()
{
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return t.QuadPart;
}

inline double usec(const counter_t& dt)
{
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    const counter_t CPS = f.QuadPart;
    return 1000000 * double(dt) / CPS;
}

#else // linux or cygwin

#include <sys/time.h>

typedef unsigned long long counter_t;

inline counter_t get_counter()
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    const counter_t s = tv.tv_sec;
    const counter_t us = 1000000*s + tv.tv_usec;
    return us;
}

inline double usec(const counter_t& dt)
{
    return double(dt);
}

#endif // _MSC_VER || __WINNT__

#include "msleep.h"

#endif // stopwatch_h
