#pragma once

#include <c10/macros/Macros.h>

// bool switch
#define PT_DISPATCH_BOOL_NAME(NAME, VAL, ...) \
    if (!(VAL)) {                             \
        static const bool NAME = false;       \
        __VA_ARGS__();                        \
    } else {                                  \
        static const bool NAME = true;        \
        __VA_ARGS__();                        \
    }

#define PT_DISPATCH_BOOL(ARG1, ...) \
    PT_DISPATCH_BOOL_NAME(ARG1, ARG1, __VA_ARGS__)

// index type
#define PT_DISPATCH_INDEX_TYPE(N_KERNELS, ...) \
    if (((int64_t)N_KERNELS) > (1 << 31)) {    \
        using index_t = int64_t;               \
        __VA_ARGS__();                         \
    }                                          \
    else {                                     \
        using index_t = int;                   \
        __VA_ARGS__();                         \
    }

#define PT_DISPATCH_INDEX_TYPE_CPU(N_KERNELS, ...) \
    using index_t = int64_t;                       \
    __VA_ARGS__();                                 \

#define PT_DISPATCH_INDEX_TYPE_CUDA(N_KERNELS, ...) \
    if (((int64_t)N_KERNELS) > (1 << 31)) {         \
        using index_t = int64_t;                    \
        __VA_ARGS__();                              \
    }                                               \
    else {                                          \
        using index_t = int;                        \
        __VA_ARGS__();                              \
    }

#define PT_DISPATCH_INDEX_TYPE_DEVICE(N_KERNELS, DEVICE, ...) \
C10_CONCATENATE(PT_DISPATCH_INDEX_TYPE_, DEVICE)(N_KERNELS, __VA_ARGS__)
