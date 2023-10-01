#pragma once

// index type
#define PT_DISPATCH_INDEX_TYPE(N_KERNELS, ...)      \
  if (((int64_t)N_KERNELS) > (1 << 31)) {           \
    using index_t = int64_t;                        \
    __VA_ARGS__();                                  \
  }                                                 \
  else {                                            \
    using index_t = int;                            \
    __VA_ARGS__();                                  \
  }

#define PT_DISPATCH_INDEX_TYPE2(N_KERNELS1, N_KERNELS2, ...)                      \
  if (((int64_t)N_KERNELS1) > (1 << 31) || ((int64_t)N_KERNELS2) > (1 << 31)) {   \
    using index_t = int64_t;                                                      \
    __VA_ARGS__();                                                                \
  }                                                                               \
  else {                                                                          \
    using index_t = int;                                                          \
    __VA_ARGS__();                                                                \
  }
