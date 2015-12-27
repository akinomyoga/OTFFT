/******************************************************************************
*  Copyright (c) 2015 OK Ojisan(Takuya OKAHISA)
*  Released under the MIT license
*  http://opensource.org/licenses/mit-license.php
*******************************************************************************/

#ifndef msleep_h
#define msleep_h

#ifdef _MSC_VER
#include <windows.h>
static inline void sleep(int n) { Sleep(1000 * n); }
static inline void msleep(int n) { Sleep(n); }
#elif defined(__WINNT__)
#include <windows.h>
static inline void sleep(int n) { Sleep(1000 * n); }
static inline void msleep(int n) { Sleep(n); }
#else
#include <unistd.h>
static inline void msleep(int n) { usleep(1000 * n); }
#endif

#endif // msleep_h
