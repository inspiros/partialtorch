#pragma once
#include <cmath>

#if defined(_MSC_VER)
#define CPU_1D_KERNEL_LOOP_BETWEEN_T(i, start, end, index_t)      \
__pragma(omp parallel for)                                        \
    for (index_t i = start; i < end; ++i)
#else
#define CPU_1D_KERNEL_LOOP_BETWEEN_T(i, start, end, index_t)      \
_Pragma("omp parallel for")                                       \
    for (index_t i = start; i < end; ++i)
#endif

#define CPU_1D_KERNEL_LOOP_T(i, n, index_t)     \
    CPU_1D_KERNEL_LOOP_BETWEEN_T(i, 0, n, index_t)

#define CPU_1D_KERNEL_LOOP(i, n)     \
    CPU_1D_KERNEL_LOOP_T(i, n, int)

#if defined(_MSC_VER)
#define __forceinline__ __forceinline
#elif defined(__GNUC__) && !defined(__clang__)
#define __forceinline__ __attribute__((always_inline)) inline
#else
#define __forceinline__ inline
#endif

using std::min;
using std::max;
using std::clamp;
using std::ceil;
using std::floor;
