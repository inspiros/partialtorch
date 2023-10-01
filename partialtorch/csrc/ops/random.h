#pragma once

#include <ATen/ATen.h>

#include "../MaskedPair.h"
#include "../macros.h"
#include "utils/torch_library_utils.h"

namespace partialtorch {
    namespace ops {
        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> rand_mask(
                const at::Tensor &data,
                double p,
                const c10::optional<at::Scalar> &mask_value = {},
                c10::optional<at::Generator> generator = c10::nullopt);

        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> rand_mask(
                const at::Tensor &data,
                const at::Tensor &p,
                const c10::optional<at::Scalar> &mask_value = {},
                c10::optional<at::Generator> generator = c10::nullopt);

        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> rand_mask(
                const_intrusive_ptr_arg_t<TensorMaskedPair> pair,
                double p,
                const c10::optional<at::Scalar> &mask_value = {},
                c10::optional<at::Generator> generator = c10::nullopt);

        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> rand_mask(
                const_intrusive_ptr_arg_t<TensorMaskedPair> pair,
                const at::Tensor &p,
                const c10::optional<at::Scalar> &mask_value = {},
                c10::optional<at::Generator> generator = c10::nullopt);

        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> rand_mask_(
                const at::Tensor &data,
                double p,
                const c10::optional<at::Scalar> &mask_value = {},
                c10::optional<at::Generator> generator = c10::nullopt);

        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> rand_mask_(
                const at::Tensor &data,
                const at::Tensor &p,
                const c10::optional<at::Scalar> &mask_value = {},
                c10::optional<at::Generator> generator = c10::nullopt);

        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> rand_mask_(
                intrusive_ptr_arg_t<TensorMaskedPair> pair,
                double p,
                const c10::optional<at::Scalar> &mask_value = {},
                c10::optional<at::Generator> generator = c10::nullopt);

        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> rand_mask_(
                intrusive_ptr_arg_t<TensorMaskedPair> pair,
                const at::Tensor &p,
                const c10::optional<at::Scalar> &mask_value = {},
                c10::optional<at::Generator> generator = c10::nullopt);
    }
}
