#pragma once

#include <ATen/ATen.h>

#include "../MaskedPair.h"
#include "../macros.h"
#include "utils/torch_library_utils.h"

namespace partialtorch {
    namespace ops {
        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> partial_einsum(
                c10::string_view equation,
                TensorMaskedPairIntrusivePtrList tensors,
                at::OptionalIntArrayRef path = c10::nullopt);
    }
}
