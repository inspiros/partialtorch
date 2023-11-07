#pragma once

#include <ATen/ATen.h>

#include "../MaskedPair.h"
#include "../macros.h"
#include "utils/torch_library_utils.h"

namespace partialtorch {
    namespace ops {
        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> pad(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                c10::SymIntArrayRef pad,
                c10::string_view mode = "constant",
                c10::optional<double> value = c10::nullopt,
                c10::string_view mask_mode = "constant",
                c10::optional<bool> mask_value = c10::nullopt);
    }
}
