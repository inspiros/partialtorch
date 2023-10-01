#pragma once

#include <ATen/ATen.h>

namespace partialtorch {
    namespace ops {
        namespace utils {
            template<typename T>
            C10_ALWAYS_INLINE c10::SymIntArrayRef to_sym_int(const T &input) {
                if constexpr (std::is_same_v<T, c10::SymIntArrayRef>)
                    return input;
                else if constexpr (std::is_same_v<T, c10::IntArrayRef>)
                    return c10::fromIntArrayRefSlow(input);
                else
                    return c10::SymIntArrayRef(input);
            }
        }
    }
}
