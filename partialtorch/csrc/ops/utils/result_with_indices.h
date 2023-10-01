#pragma once

#include <ATen/ATen.h>

#include "../../MaskedPair.h"

namespace partialtorch {
    namespace ops {
        // msvc's preprocessor cannot unfold this
        using tensor_with_indices = std::tuple<at::Tensor, at::Tensor>;
        using masked_pair_with_indices = std::tuple<c10::intrusive_ptr<partialtorch::TensorMaskedPair>, at::Tensor>;
    }
}
