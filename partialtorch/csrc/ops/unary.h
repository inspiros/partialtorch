#pragma once

#include <ATen/ATen.h>

#include "../MaskedPair.h"
#include "../macros.h"
#include "utils/torch_library_utils.h"

// ops declaration macros
#define PT_DECLARE_UNARY_OP(NAME, SELF_T) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME( \
    SELF_T self);

#define PT_DECLARE_UNARY_OP_WITH(NAME, SELF_T, ARG1) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME( \
    SELF_T self, ARG1);

#define PT_DECLARE_UNARY_OP_WITH2(NAME, SELF_T, ARG1, ARG2) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME( \
    SELF_T self, ARG1, ARG2);

#define PT_DECLARE_UNARY_OP_WITH3(NAME, SELF_T, ARG1, ARG2, ARG3) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME(       \
    SELF_T self, ARG1, ARG2, ARG3);

#define PT_DECLARE_UNARY_OP_WITH4(NAME, SELF_T, ARG1, ARG2, ARG3, ARG4) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME(             \
    SELF_T self, ARG1, ARG2, ARG3, ARG4);

#define PT_DECLARE_UNARY_OP_(NAME, SELF_T) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME##_( \
    SELF_T self);

#define PT_DECLARE_UNARY_OP__WITH(NAME, SELF_T, ARG1) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME##_( \
    SELF_T self, ARG1);

#define PT_DECLARE_UNARY_OP__WITH2(NAME, SELF_T, ARG1, ARG2) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME##_( \
    SELF_T self, ARG1, ARG2);

#define PT_DECLARE_UNARY_OP__WITH3(NAME, SELF_T, ARG1, ARG2, ARG3) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME##_(     \
    SELF_T self, ARG1, ARG2, ARG3);

#define PT_DECLARE_UNARY_OP__WITH4(NAME, SELF_T, ARG1, ARG2, ARG3, ARG4) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME##_(           \
    SELF_T self, ARG1, ARG2, ARG3, ARG4);

#define PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(NAME) \
PT_DECLARE_UNARY_OP(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>)

#define PT_DECLARE_UNARY_OPS__FORALL_TENSOR_OVERLOADS(NAME) \
PT_DECLARE_UNARY_OP_(NAME, intrusive_ptr_arg_t<TensorMaskedPair>)

#define PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(NAME) \
PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(NAME)                  \
PT_DECLARE_UNARY_OPS__FORALL_TENSOR_OVERLOADS(NAME)

#define PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, ARG1) \
PT_DECLARE_UNARY_OP_WITH(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1)

#define PT_DECLARE_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, ARG1) \
PT_DECLARE_UNARY_OP__WITH(NAME, intrusive_ptr_arg_t<TensorMaskedPair>, ARG1)

#define PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, ARG1) \
PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, ARG1)                  \
PT_DECLARE_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, ARG1)

#define PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, ARG1, ARG2) \
PT_DECLARE_UNARY_OP_WITH2(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1, ARG2)

#define PT_DECLARE_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, ARG1, ARG2) \
PT_DECLARE_UNARY_OP__WITH2(NAME, intrusive_ptr_arg_t<TensorMaskedPair>, ARG1, ARG2)

#define PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, ARG1, ARG2) \
PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, ARG1, ARG2)                  \
PT_DECLARE_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, ARG1, ARG2)

#define PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH3(NAME, ARG1, ARG2, ARG3) \
PT_DECLARE_UNARY_OP_WITH3(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1, ARG2, ARG3)

#define PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH4(NAME, ARG1, ARG2, ARG3, ARG4) \
PT_DECLARE_UNARY_OP_WITH4(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1, ARG2, ARG3, ARG4)

#define PT_DECLARE_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH3(NAME, ARG1, ARG2, ARG3) \
PT_DECLARE_UNARY_OP__WITH3(NAME, intrusive_ptr_arg_t<TensorMaskedPair>, ARG1, ARG2, ARG3)

#define PT_DECLARE_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH4(NAME, ARG1, ARG2, ARG3, ARG4) \
PT_DECLARE_UNARY_OP__WITH4(NAME, intrusive_ptr_arg_t<TensorMaskedPair>, ARG1, ARG2, ARG3, ARG4)

#define PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH3(NAME, ARG1, ARG2, ARG3) \
PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH3(NAME, ARG1, ARG2, ARG3)                  \
PT_DECLARE_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH3(NAME, ARG1, ARG2, ARG3)

#define PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH4(NAME, ARG1, ARG2, ARG3, ARG4) \
PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH4(NAME, ARG1, ARG2, ARG3, ARG4)                  \
PT_DECLARE_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH4(NAME, ARG1, ARG2, ARG3, ARG4)

namespace partialtorch {
    namespace ops {
        // custom ops
        PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(identity)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                fill_masked, const at::Scalar &value)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                fill_masked, const at::Tensor &value)

        PARTIALTORCH_API at::Tensor to_tensor(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                const at::Scalar &value);

        PARTIALTORCH_API at::Tensor index_non_masked(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self);

        // torch ops
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(abs)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(absolute)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(acos)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(arccos)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(acosh)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(arccosh)
        PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(alias)
        PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(angle)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(asin)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(arcsin)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(asinh)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(arcsinh)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(atan)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(arctan)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(atanh)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(arctanh)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(bitwise_not)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(ceil)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                clamp,
                const c10::optional<at::Scalar> &min = c10::nullopt,
                const c10::optional<at::Scalar> &max = c10::nullopt)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                clamp,
                const c10::optional<at::Tensor> &min = c10::nullopt,
                const c10::optional<at::Tensor> &max = c10::nullopt)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                clip,
                const c10::optional<at::Scalar> &min = c10::nullopt,
                const c10::optional<at::Scalar> &max = c10::nullopt)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                clip,
                const c10::optional<at::Tensor> &min = c10::nullopt,
                const c10::optional<at::Tensor> &max = c10::nullopt)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(conj_physical)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(cos)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(cosh)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(deg2rad)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(rad2deg)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                dropout, double p, bool train)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                feature_dropout, double p, bool train)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                alpha_dropout, double p, bool train)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                feature_alpha_dropout, double p, bool train)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(erf)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(erfc)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(erfinv)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(exp)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(exp2)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(expm1)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(fix)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(floor)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(frac)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(lgamma)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(digamma)

        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> polygamma(
                int64_t n, const_intrusive_ptr_arg_t<TensorMaskedPair> self);

        PT_DECLARE_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                polygamma, int64_t n)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(log)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(log10)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(log1p)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(log2)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                logit, const c10::optional<double> eps = c10::nullopt)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(i0)
        PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(isnan)
        PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(isreal)
        PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(isfinite)
        PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(isinf)
        PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(isposinf)
        PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(isneginf)
        PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(matrix_exp)
        PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                matrix_power,
                int64_t n)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH3(
                nan_to_num,
                c10::optional<double> nan = c10::nullopt,
                c10::optional<double> posinf = c10::nullopt,
                c10::optional<double> neginf = c10::nullopt)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(neg)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(negative)
        PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(positive)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(reciprocal)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(round)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(rsqrt)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(sign)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(sgn)
        PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(signbit)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(sin)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(sinc)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(sinh)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(sqrt)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(square)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(tan)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(tanh)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                hardtanh,
                const at::Scalar &min_val = -1,
                const at::Scalar &max_val = 1)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                threshold,
                const at::Scalar &threshold,
                const at::Scalar &value)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(trunc)
        // activations
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(relu)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(relu6)
        PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                glu,
                int64_t dim = -1)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH3(
                elu,
                const at::Scalar &alpha = 1,
                const at::Scalar &scale = 1,
                const at::Scalar &input_scale = 1)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(selu)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                celu,
                const at::Scalar &alpha = 1)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                leaky_relu,
                const at::Scalar &negative_slope = 0.01)
        PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                prelu,
                const at::Tensor &weight)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH4(
                rrelu,
                const at::Scalar &lower = 0.125,
                const at::Scalar &upper = 0.3333333333333333,
                bool training = false,
                c10::optional<at::Generator> generator = c10::nullopt)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                gelu,
                c10::string_view approximate = "none")
        PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                hardshrink,
                const at::Scalar &lambd = 0.5)
        PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                softshrink,
                const at::Scalar &lambd = 0.5)
        PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                softplus,
                const at::Scalar &beta = 1,
                const at::Scalar &threshold = 20)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(sigmoid)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(hardsigmoid)
        PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(log_sigmoid)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(silu)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(mish)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(hardswish)

        // torch.nn.functional
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                dropout1d, double p, bool train)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                dropout2d, double p, bool train)
        PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                dropout3d, double p, bool train)
        PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(logsigmoid)
        PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(softsign)
        PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(tanhshrink)

        // properties
        PARTIALTORCH_API int64_t dim(const_intrusive_ptr_arg_t<TensorMaskedPair> self);

        PARTIALTORCH_API int64_t dense_dim(const_intrusive_ptr_arg_t<TensorMaskedPair> self);

        PARTIALTORCH_API int64_t size(const_intrusive_ptr_arg_t<TensorMaskedPair> self, int64_t dim);

        PT_DECLARE_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                requires_grad, bool requires_grad = true)
    }
}

#undef PT_DECLARE_UNARY_OP
#undef PT_DECLARE_UNARY_OP_WITH
#undef PT_DECLARE_UNARY_OP_WITH2
#undef PT_DECLARE_UNARY_OP_WITH3
#undef PT_DECLARE_UNARY_OP_WITH4
#undef PT_DECLARE_UNARY_OP_
#undef PT_DECLARE_UNARY_OP__WITH
#undef PT_DECLARE_UNARY_OP__WITH2
#undef PT_DECLARE_UNARY_OP__WITH3
#undef PT_DECLARE_UNARY_OP__WITH4
#undef PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS
#undef PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH
#undef PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2
#undef PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH3
#undef PT_DECLARE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH4
#undef PT_DECLARE_UNARY_OPS__FORALL_TENSOR_OVERLOADS
#undef PT_DECLARE_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH
#undef PT_DECLARE_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH2
#undef PT_DECLARE_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH3
#undef PT_DECLARE_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH4
#undef PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS
#undef PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH
#undef PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2
#undef PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH3
#undef PT_DECLARE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH4
