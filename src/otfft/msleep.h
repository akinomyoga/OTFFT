/******************************************************************************
*  Copyright (c) 2015 OK Ojisan(Takuya OKAHISA)
*  Released under the MIT license
*  http://opensource.org/licenses/mit-license.php
*******************************************************************************/

#ifndef msleep_h
#define msleep_h

#include <chrono>
#include <thread>

static inline void sleep(int n)
{
    std::this_thread::sleep_for(std::chrono::seconds(n));
}

static inline void msleep(int n)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(n));
}

#endif // msleep_h
