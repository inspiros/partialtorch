#pragma once

#ifdef _WIN32
#if defined(partialtorch_EXPORTS)
#define PARTIALTORCH_API __declspec(dllexport)
#else
#define PARTIALTORCH_API __declspec(dllimport)
#endif
#else
#define PARTIALTORCH_API
#endif

#if (defined __cpp_inline_variables) || __cplusplus >= 201703L
#define PARTIALTORCH_INLINE_VARIABLE inline
#else
#ifdef _MSC_VER
#define PARTIALTORCH_INLINE_VARIABLE __declspec(selectany)
#define HINT_MSVC_LINKER_INCLUDE_SYMBOL
#else
#define PARTIALTORCH_INLINE_VARIABLE __attribute__((weak))
#endif
#endif
