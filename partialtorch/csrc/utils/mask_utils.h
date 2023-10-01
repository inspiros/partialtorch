#pragma once

#include <ATen/ATen.h>

namespace partialtorch {
    C10_ALWAYS_INLINE void validate_masked_pair(
            const at::Tensor &data,
            const c10::optional<at::Tensor> &mask) {
        if (mask.has_value()) {
            TORCH_CHECK_VALUE(mask->defined(),
                              "mask must be defined tensor.")
            TORCH_CHECK_VALUE(mask->sizes() == data.sizes(),
                              "data and mask shapes do not match. got data.shape=",
                              data.sizes(),
                              ", mask.shape=",
                              mask->sizes())
            TORCH_CHECK_VALUE(mask->scalar_type() == at::kBool,
                              "mask must be bool tensor. got mask.dtype=",
                              mask->scalar_type())
        }
    }
}
