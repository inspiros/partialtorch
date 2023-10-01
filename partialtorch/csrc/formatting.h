#pragma once

#include <ostream>
#include <string>

#include "MaskedPair.h"
#include "macros.h"

namespace partialtorch {
    PARTIALTORCH_API std::ostream &print(std::ostream &stream, const MaskedPair<at::Tensor> &p, int64_t linesize = 80);

    PARTIALTORCH_API void print(const MaskedPair<at::Tensor> &p, int64_t linesize = 80);

    PARTIALTORCH_API inline std::ostream &operator<<(std::ostream &stream, const MaskedPair<at::Tensor> &self) {
        return print(stream, self);
    }
}
