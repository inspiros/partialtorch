#pragma once

#include <ATen/ATen.h>

#include "../MaskedPair.h"
#include "../macros.h"
#include "utils/torch_library_utils.h"

// ops declaration macros
#define PT_DECLARE_REDUCTION_OP(NAME, SELF_T) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME( \
    SELF_T self);

#define PT_DECLARE_REDUCTION_OP_WITH(NAME, SELF_T, ARG1) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME( \
    SELF_T self, ARG1);

#define PT_DECLARE_REDUCTION_OP_WITH2(NAME, SELF_T, ARG1, ARG2) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME( \
    SELF_T self, ARG1, ARG2);

#define PT_DECLARE_REDUCTION_DIM_OP(NAME, SELF_T, DIM) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME( \
    SELF_T self, DIM);

#define PT_DECLARE_REDUCTION_DIM_OP_WITH(NAME, SELF_T, DIM, ARG1) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME(\
    SELF_T self, DIM, ARG1);

#define PT_DECLARE_REDUCTION_KEEPDIM_OP(NAME, SELF_T, DIM) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME( \
    SELF_T self, DIM, bool keepdim = false);

#define PT_DECLARE_REDUCTION_KEEPDIM_OP_WITH(NAME, SELF_T, DIM, ARG1) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME(    \
    SELF_T self, DIM, bool keepdim = false, ARG1);

#define PT_DECLARE_REDUCTION_KEEPDIM_OP_WITH2(NAME, SELF_T, DIM, ARG1, ARG2) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME(    \
    SELF_T self, DIM, bool keepdim = false, ARG1, ARG2);

#define PT_DECLARE_REDUCTION_DIM_OP_(NAME, SELF_T, DIM) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME##_( \
    SELF_T self, DIM);

#define PT_DECLARE_REDUCTION_DIM_OP__WITH(NAME, SELF_T, DIM, ARG1) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME##_( \
    SELF_T self, DIM, ARG1);

#define PT_DECLARE_REDUCTION_INDICES_DIM_OP(NAME, SELF_T, DIM) \
PARTIALTORCH_API std::tuple<c10::intrusive_ptr<TensorMaskedPair>, at::Tensor> NAME( \
    SELF_T self, DIM);

#define PT_DECLARE_REDUCTION_INDICES_KEEPDIM_OP(NAME, SELF_T, DIM) \
PARTIALTORCH_API std::tuple<c10::intrusive_ptr<TensorMaskedPair>, at::Tensor> NAME( \
    SELF_T self, DIM, bool keepdim = false);

#define PT_DECLARE_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS(NAME) \
PT_DECLARE_REDUCTION_OP(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>)

#define PT_DECLARE_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, ARG1) \
PT_DECLARE_REDUCTION_OP_WITH(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1)

#define PT_DECLARE_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, ARG1, ARG2) \
PT_DECLARE_REDUCTION_OP_WITH2(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1, ARG2)

#define PT_DECLARE_REDUCTION_DIM_OPS_FORALL_TENSOR_OVERLOADS(NAME, DIM) \
PT_DECLARE_REDUCTION_DIM_OP(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, DIM)

#define PT_DECLARE_REDUCTION_DIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, DIM, ARG1) \
PT_DECLARE_REDUCTION_DIM_OP_WITH(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, DIM, ARG1)

#define PT_DECLARE_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(NAME, DIM) \
PT_DECLARE_REDUCTION_KEEPDIM_OP(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, DIM)

#define PT_DECLARE_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, DIM, ARG1) \
PT_DECLARE_REDUCTION_KEEPDIM_OP_WITH(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, DIM, ARG1)

#define PT_DECLARE_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, DIM, ARG1, ARG2) \
PT_DECLARE_REDUCTION_KEEPDIM_OP_WITH2(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, DIM, ARG1, ARG2)

#define PT_DECLARE_REDUCTION_DIM_OPS__FORALL_TENSOR_OVERLOADS(NAME, DIM) \
PT_DECLARE_REDUCTION_DIM_OP_(NAME, intrusive_ptr_arg_t<TensorMaskedPair>, DIM)

#define PT_DECLARE_REDUCTION_DIM_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, DIM, ARG1) \
PT_DECLARE_REDUCTION_DIM_OP__WITH(NAME, intrusive_ptr_arg_t<TensorMaskedPair>, DIM, ARG1)

#define PT_DECLARE_REDUCTION_INDICES_DIM_OPS_FORALL_TENSOR_OVERLOADS(NAME, DIM) \
PT_DECLARE_REDUCTION_INDICES_DIM_OP(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, DIM)

#define PT_DECLARE_REDUCTION_INDICES_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(NAME, DIM) \
PT_DECLARE_REDUCTION_INDICES_KEEPDIM_OP(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, DIM)

#define PT_DECLARE_REDUCTION_AND_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(NAME, DIM) \
PT_DECLARE_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS(NAME)                          \
PT_DECLARE_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(NAME, DIM)

#define PT_DECLARE_REDUCTION_AND_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, DIM, ARG1) \
PT_DECLARE_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, ARG1)                          \
PT_DECLARE_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, DIM, ARG1)

#define PT_DECLARE_REDUCTION_AND_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, DIM, ARG1, ARG2) \
PT_DECLARE_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, ARG1, ARG2)                          \
PT_DECLARE_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, DIM, ARG1, ARG2)

#define PT_DECLARE_REDUCTION_DIM_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(NAME, DIM) \
PT_DECLARE_REDUCTION_DIM_OPS_FORALL_TENSOR_OVERLOADS(NAME, DIM)                  \
PT_DECLARE_REDUCTION_DIM_OPS__FORALL_TENSOR_OVERLOADS(NAME, DIM)

#define PT_DECLARE_REDUCTION_DIM_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, DIM, ARG1) \
PT_DECLARE_REDUCTION_DIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, DIM, ARG1)                  \
PT_DECLARE_REDUCTION_DIM_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, DIM, ARG1)

namespace partialtorch {
    namespace ops {
        // logical
        PT_DECLARE_REDUCTION_AND_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(
                all, at::OptionalIntArrayRef dim)

        PT_DECLARE_REDUCTION_AND_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(
                any, at::OptionalIntArrayRef dim)

        // arithmetics
        PT_DECLARE_REDUCTION_AND_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                sum, at::OptionalIntArrayRef dim,
                c10::optional<at::ScalarType> dtype = {}, bool scaled = false)

        PT_DECLARE_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                nansum, at::OptionalIntArrayRef dim,
                c10::optional<at::ScalarType> dtype = {}, bool scaled = false)

        PT_DECLARE_REDUCTION_AND_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                prod, at::OptionalIntArrayRef dim, c10::optional<at::ScalarType> dtype = {})

        PT_DECLARE_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(
                logsumexp, at::IntArrayRef dim)

        PT_DECLARE_REDUCTION_DIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                softmax, int64_t dim, c10::optional<at::ScalarType> dtype = {})

        PT_DECLARE_REDUCTION_DIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                log_softmax, int64_t dim, c10::optional<at::ScalarType> dtype = {})

        PT_DECLARE_REDUCTION_DIM_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                cumsum, int64_t dim, c10::optional<at::ScalarType> dtype = {})

        PT_DECLARE_REDUCTION_DIM_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                cumprod, int64_t dim, c10::optional<at::ScalarType> dtype = {})

        PT_DECLARE_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                trace, bool scaled = false)

        // statistics
        PT_DECLARE_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                mean, c10::optional<at::ScalarType> dtype = {})

        PT_DECLARE_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                mean, at::OptionalIntArrayRef dim = {}, c10::optional<at::ScalarType> dtype = {})

        PT_DECLARE_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                nanmean, at::OptionalIntArrayRef dim = {}, c10::optional<at::ScalarType> dtype = {})

        PT_DECLARE_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS(median)

        PT_DECLARE_REDUCTION_INDICES_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(
                median, int64_t dim)

        PT_DECLARE_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS(nanmedian)

        PT_DECLARE_REDUCTION_INDICES_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(
                nanmedian, int64_t dim)

        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> norm(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                const c10::optional<at::Scalar> &p = {},
                at::IntArrayRef dim = {},
                bool keepdim = false,
                c10::optional<at::ScalarType> dtype = {});

        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> var(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                bool unbiased = true);

        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> var(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                at::OptionalIntArrayRef dim,
                bool unbiased = true,
                bool keepdim = false);

        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> var(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                at::OptionalIntArrayRef dim = {},
                c10::optional<int64_t> correction = {},
                bool keepdim = false);

        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> std(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                bool unbiased = true);

        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> std(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                at::OptionalIntArrayRef dim,
                bool unbiased = true,
                bool keepdim = false);

        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> std(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                at::OptionalIntArrayRef dim = {},
                c10::optional<int64_t> correction = {},
                bool keepdim = false);

        // min max
        PT_DECLARE_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS(min)

        PT_DECLARE_REDUCTION_INDICES_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(
                min, int64_t dim)

        PT_DECLARE_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS(max)

        PT_DECLARE_REDUCTION_INDICES_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(
                max, int64_t dim)

        PT_DECLARE_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(
                amin, at::IntArrayRef dim = {})

        PT_DECLARE_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(
                amax, at::IntArrayRef dim = {})

        PARTIALTORCH_API at::Tensor argmin(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                c10::optional<int64_t> dim = {},
                bool keepdim = false);

        PARTIALTORCH_API at::Tensor argmax(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                c10::optional<int64_t> dim = {},
                bool keepdim = false);

        PT_DECLARE_REDUCTION_INDICES_DIM_OPS_FORALL_TENSOR_OVERLOADS(
                cummin, int64_t dim)

        PT_DECLARE_REDUCTION_INDICES_DIM_OPS_FORALL_TENSOR_OVERLOADS(
                cummax, int64_t dim)

        // torch.nn.functional
        PT_DECLARE_REDUCTION_DIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                softmin, int64_t dim, c10::optional<at::ScalarType> dtype = {})

        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> normalize(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                const c10::optional<at::Scalar> &p = 2,
                int64_t dim = 1,
                double eps = 1e-12);
    }
}

#undef PT_DECLARE_REDUCTION_OP
#undef PT_DECLARE_REDUCTION_OP_WITH
#undef PT_DECLARE_REDUCTION_OP_WITH2
#undef PT_DECLARE_REDUCTION_DIM_OP
#undef PT_DECLARE_REDUCTION_DIM_OP_WITH
#undef PT_DECLARE_REDUCTION_KEEPDIM_OP
#undef PT_DECLARE_REDUCTION_KEEPDIM_OP_WITH
#undef PT_DECLARE_REDUCTION_KEEPDIM_OP_WITH2
#undef PT_DECLARE_REDUCTION_DIM_OP_
#undef PT_DECLARE_REDUCTION_DIM_OP__WITH
#undef PT_DECLARE_REDUCTION_INDICES_DIM_OP
#undef PT_DECLARE_REDUCTION_INDICES_KEEPDIM_OP
#undef PT_DECLARE_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS
#undef PT_DECLARE_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS_WITH
#undef PT_DECLARE_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS_WITH2
#undef PT_DECLARE_REDUCTION_DIM_OPS_FORALL_TENSOR_OVERLOADS
#undef PT_DECLARE_REDUCTION_DIM_OPS_FORALL_TENSOR_OVERLOADS_WITH
#undef PT_DECLARE_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS
#undef PT_DECLARE_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH
#undef PT_DECLARE_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH2
#undef PT_DECLARE_REDUCTION_DIM_OPS__FORALL_TENSOR_OVERLOADS
#undef PT_DECLARE_REDUCTION_DIM_OPS__FORALL_TENSOR_OVERLOADS_WITH
#undef PT_DECLARE_REDUCTION_INDICES_DIM_OPS_FORALL_TENSOR_OVERLOADS
#undef PT_DECLARE_REDUCTION_INDICES_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS
#undef PT_DECLARE_REDUCTION_AND_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS
#undef PT_DECLARE_REDUCTION_AND_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH
#undef PT_DECLARE_REDUCTION_AND_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH2
#undef PT_DECLARE_REDUCTION_DIM_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS
#undef PT_DECLARE_REDUCTION_DIM_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH
