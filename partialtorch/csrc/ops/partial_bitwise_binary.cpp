#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <torch/library.h>

#include "../MaskedPair.h"
#include "izero_div.h"
#include "utils/mask_utils.h"
#include "utils/fill_identity.h"
#include "utils/schema_utils.h"
#include "utils/torch_library_utils.h"

// ~~~~~ ops definition macros ~~~~~
#define PT_DEFINE_BINARY_OP(NAME, IMPL_OP, IMPL_OP_, IDENTITY_OP, SELF_T, OTHER_T) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                         \
    SELF_T self,                                                                   \
    OTHER_T other) {                                                               \
    static constexpr auto op = IMPL_OP;                                            \
    static constexpr auto op_ = IMPL_OP_;                                          \
    static constexpr auto fill_identity_op = IDENTITY_OP;                          \
    return impl::partial_bitwise_binary_impl(op, op_, fill_identity_op, self, other); \
}

#define PT_DEFINE_BINARY_OP_WITH(NAME, IMPL_OP, IMPL_OP_, IDENTITY_OP, SELF_T, OTHER_T, ARG1_T, ARG1_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                                 \
    SELF_T self,                                                                                           \
    OTHER_T other,                                                                                         \
    ARG1_T ARG1_NAME) {                                                                                    \
    static constexpr auto op = IMPL_OP;                                                                    \
    static constexpr auto op_ = IMPL_OP_;                                                                  \
    static constexpr auto fill_identity_op = IDENTITY_OP;                                                  \
    return impl::partial_bitwise_binary_impl(op, op_, fill_identity_op, self, other, ARG1_NAME);           \
}

#define PT_DEFINE_BINARY_OP_WITH2(NAME, IMPL_OP, IMPL_OP_, IDENTITY_OP, SELF_T, OTHER_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                                                     \
    SELF_T self,                                                                                                               \
    OTHER_T other,                                                                                                             \
    ARG1_T ARG1_NAME,                                                                                                          \
    ARG2_T ARG2_NAME) {                                                                                                        \
    static constexpr auto op = IMPL_OP;                                                                                        \
    static constexpr auto op_ = IMPL_OP_;                                                                                      \
    static constexpr auto fill_identity_op = IDENTITY_OP;                                                                      \
    return impl::partial_bitwise_binary_impl(op, op_, fill_identity_op, self, other, ARG1_NAME, ARG2_NAME);                    \
}

#define PT_DEFINE_BINARY_OP_(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, SELF_T, OTHER_T) \
c10::intrusive_ptr<TensorMaskedPair> NAME##_(                                            \
    SELF_T self,                                                                         \
    OTHER_T other) {                                                                     \
    static constexpr auto op_ = IMPL_OP_;                                                \
    static constexpr auto fill_identity_op = IDENTITY_OP;                                \
    static constexpr auto fill_identity_op_ = IDENTITY_OP_;                              \
    return impl::partial_bitwise_binary__impl(op_, fill_identity_op, fill_identity_op_, self, other); \
}

#define PT_DEFINE_BINARY_OP__WITH(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, SELF_T, OTHER_T, ARG1_T, ARG1_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME##_(                                                                    \
    SELF_T self,                                                                                                 \
    OTHER_T other,                                                                                               \
    ARG1_T ARG1_NAME) {                                                                                          \
    static constexpr auto op_ = IMPL_OP_;                                                                        \
    static constexpr auto fill_identity_op = IDENTITY_OP;                                                        \
    static constexpr auto fill_identity_op_ = IDENTITY_OP_;                                                      \
    return impl::partial_bitwise_binary__impl(op_, fill_identity_op, fill_identity_op_, self, other, ARG1_NAME); \
}

#define PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP) \
PT_DEFINE_BINARY_OP(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DEFINE_BINARY_OP(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &)                          \
PT_DEFINE_BINARY_OP(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>)                          \
PT_DEFINE_BINARY_OP(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, const at::Tensor &, const at::Tensor &)

#define PT_DEFINE_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(NAME, TENSOR_OP, TENSOR_OP_, SCALAR_OP, SCALAR_OP_, IDENTITY_OP) \
PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP)                                                 \
PT_DEFINE_BINARY_OP(NAME, SCALAR_OP, SCALAR_OP_, IDENTITY_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Scalar &)         \
PT_DEFINE_BINARY_OP(NAME, SCALAR_OP, SCALAR_OP_, IDENTITY_OP, const at::Tensor &, const at::Scalar &)

#define PT_DEFINE_BINARY_OPS_FORALL_TENSOR_AND_SCALAR_OVERLOADS(NAME, TENSOR_OP, TENSOR_OP_, SCALAR_OP, SCALAR_OP_, IDENTITY_OP) \
PT_DEFINE_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(NAME, TENSOR_OP, TENSOR_OP_, SCALAR_OP, SCALAR_OP_, IDENTITY_OP)   \
PT_DEFINE_BINARY_OP(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, const at::Scalar &, const_intrusive_ptr_arg_t<TensorMaskedPair>)   \
PT_DEFINE_BINARY_OP(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, const at::Scalar &, const at::Tensor &)

#define PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, ARG1_T, ARG1_NAME) \
PT_DEFINE_BINARY_OP_WITH(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME) \
PT_DEFINE_BINARY_OP_WITH(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1_T, ARG1_NAME)                          \
PT_DEFINE_BINARY_OP_WITH(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME)                          \
PT_DEFINE_BINARY_OP_WITH(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, const at::Tensor &, const at::Tensor &, ARG1_T, ARG1_NAME)

#define PT_DEFINE_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH(NAME, TENSOR_OP, TENSOR_OP_, SCALAR_OP, SCALAR_OP_, IDENTITY_OP, ARG1_T, ARG1_NAME) \
PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, ARG1_T, ARG1_NAME)                                                 \
PT_DEFINE_BINARY_OP_WITH(NAME, SCALAR_OP, SCALAR_OP_, IDENTITY_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Scalar &, ARG1_T, ARG1_NAME)         \
PT_DEFINE_BINARY_OP_WITH(NAME, SCALAR_OP, SCALAR_OP_, IDENTITY_OP, const at::Tensor &, const at::Scalar &, ARG1_T, ARG1_NAME)

#define PT_DEFINE_BINARY_OPS_FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(NAME, TENSOR_OP, TENSOR_OP_, SCALAR_OP, SCALAR_OP_, IDENTITY_OP, ARG1_T, ARG1_NAME) \
PT_DEFINE_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH(NAME, TENSOR_OP, TENSOR_OP_, SCALAR_OP, SCALAR_OP_, IDENTITY_OP, ARG1_T, ARG1_NAME)   \
PT_DEFINE_BINARY_OP_WITH(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, const at::Scalar &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME)   \
PT_DEFINE_BINARY_OP_WITH(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, const at::Scalar &, const at::Tensor &, ARG1_T, ARG1_NAME)

#define PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_BINARY_OP_WITH2(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_BINARY_OP_WITH2(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                          \
PT_DEFINE_BINARY_OP_WITH2(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                          \
PT_DEFINE_BINARY_OP_WITH2(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, const at::Tensor &, const at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)

#define PT_DEFINE_BINARY_OPS__FORALL_TENSOR_OVERLOADS(NAME, TENSOR_OP_, IDENTITY_OP, IDENTITY_OP_) \
PT_DEFINE_BINARY_OP_(NAME, TENSOR_OP_, IDENTITY_OP, IDENTITY_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DEFINE_BINARY_OP_(NAME, TENSOR_OP_, IDENTITY_OP, IDENTITY_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &)                          \
PT_DEFINE_BINARY_OP_(NAME, TENSOR_OP_, IDENTITY_OP, IDENTITY_OP_, at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>)                          \
PT_DEFINE_BINARY_OP_(NAME, TENSOR_OP_, IDENTITY_OP, IDENTITY_OP_, at::Tensor &, const at::Tensor &)

#define PT_DEFINE_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(NAME, TENSOR_OP_, SCALAR_OP_, IDENTITY_OP, IDENTITY_OP_) \
PT_DEFINE_BINARY_OPS__FORALL_TENSOR_OVERLOADS(NAME, TENSOR_OP_, IDENTITY_OP, IDENTITY_OP_)                                      \
PT_DEFINE_BINARY_OP_(NAME, SCALAR_OP_, IDENTITY_OP, IDENTITY_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Scalar &)    \
PT_DEFINE_BINARY_OP_(NAME, SCALAR_OP_, IDENTITY_OP, IDENTITY_OP_, at::Tensor &, const at::Scalar &)

#define PT_DEFINE_BINARY_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(NAME, TENSOR_OP_, SCALAR_OP_, IDENTITY_OP, IDENTITY_OP_) \
PT_DEFINE_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(NAME, TENSOR_OP_, SCALAR_OP_, IDENTITY_OP, IDENTITY_OP_)   \
PT_DEFINE_BINARY_OP_(NAME, TENSOR_OP_, IDENTITY_OP, IDENTITY_OP_, const at::Scalar &, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DEFINE_BINARY_OP_(NAME, TENSOR_OP_, IDENTITY_OP, IDENTITY_OP_, const at::Scalar &, const at::Tensor &)

#define PT_DEFINE_BINARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, TENSOR_OP_, IDENTITY_OP, IDENTITY_OP_, ARG1_T, ARG1_NAME) \
PT_DEFINE_BINARY_OP__WITH(NAME, TENSOR_OP_, IDENTITY_OP, IDENTITY_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME) \
PT_DEFINE_BINARY_OP__WITH(NAME, TENSOR_OP_, IDENTITY_OP, IDENTITY_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1_T, ARG1_NAME)                          \
PT_DEFINE_BINARY_OP__WITH(NAME, TENSOR_OP_, IDENTITY_OP, IDENTITY_OP_, at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME)                          \
PT_DEFINE_BINARY_OP__WITH(NAME, TENSOR_OP_, IDENTITY_OP, IDENTITY_OP_, at::Tensor &, const at::Tensor &, ARG1_T, ARG1_NAME)

#define PT_DEFINE_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH(NAME, TENSOR_OP_, SCALAR_OP_, IDENTITY_OP, IDENTITY_OP_, ARG1_T, ARG1_NAME) \
PT_DEFINE_BINARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, TENSOR_OP_, IDENTITY_OP, IDENTITY_OP_, ARG1_T, ARG1_NAME)                                      \
PT_DEFINE_BINARY_OP__WITH(NAME, SCALAR_OP_, IDENTITY_OP, IDENTITY_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Scalar &, ARG1_T, ARG1_NAME)    \
PT_DEFINE_BINARY_OP__WITH(NAME, SCALAR_OP_, IDENTITY_OP, IDENTITY_OP_, at::Tensor &, const at::Scalar &, ARG1_T, ARG1_NAME)

#define PT_DEFINE_BINARY_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(NAME, TENSOR_OP_, SCALAR_OP_, IDENTITY_OP, IDENTITY_OP_, ARG1_T, ARG1_NAME) \
PT_DEFINE_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH(NAME, TENSOR_OP_, SCALAR_OP_, IDENTITY_OP, IDENTITY_OP_, ARG1_T, ARG1_NAME)   \
PT_DEFINE_BINARY_OP__WITH(NAME, TENSOR_OP_, IDENTITY_OP, IDENTITY_OP_, const at::Scalar &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME) \
PT_DEFINE_BINARY_OP__WITH(NAME, TENSOR_OP_, IDENTITY_OP, IDENTITY_OP_, const at::Scalar &, const at::Tensor &, ARG1_T, ARG1_NAME)

#define PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, IDENTITY_OP_) \
PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP)                                \
PT_DEFINE_BINARY_OPS__FORALL_TENSOR_OVERLOADS(NAME, TENSOR_OP_, IDENTITY_OP, IDENTITY_OP_)

#define PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(NAME, TENSOR_OP, TENSOR_OP_, SCALAR_OP, SCALAR_OP_, IDENTITY_OP, IDENTITY_OP_) \
PT_DEFINE_BINARY_OPS_FORALL_TENSOR_AND_SCALAR_OVERLOADS(NAME, TENSOR_OP, TENSOR_OP_, SCALAR_OP, SCALAR_OP_, IDENTITY_OP)                                \
PT_DEFINE_BINARY_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(NAME, TENSOR_OP_, SCALAR_OP_, IDENTITY_OP, IDENTITY_OP_)

#define PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, IDENTITY_OP_, ARG1_T, ARG1_NAME) \
PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, ARG1_T, ARG1_NAME)                                \
PT_DEFINE_BINARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, TENSOR_OP_, IDENTITY_OP, IDENTITY_OP_, ARG1_T, ARG1_NAME)

#define PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(NAME, TENSOR_OP, TENSOR_OP_, SCALAR_OP, SCALAR_OP_, IDENTITY_OP, IDENTITY_OP_, ARG1_T, ARG1_NAME) \
PT_DEFINE_BINARY_OPS_FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(NAME, TENSOR_OP, TENSOR_OP_, SCALAR_OP, SCALAR_OP_, IDENTITY_OP, ARG1_T, ARG1_NAME)                                \
PT_DEFINE_BINARY_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(NAME, TENSOR_OP_, SCALAR_OP_, IDENTITY_OP, IDENTITY_OP_, ARG1_T, ARG1_NAME)

// scaled ops
#define PT_DEFINE_SCALED_BINARY_OP_WITH(NAME, IMPL_OP, IMPL_OP_, IDENTITY_OP, SELF_T, OTHER_T, ARG1_T, ARG1_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                                        \
    SELF_T self,                                                                                                  \
    OTHER_T other,                                                                                                \
    ARG1_T ARG1_NAME,                                                                                             \
    bool scaled) {                                                                                                \
    static constexpr auto op = IMPL_OP;                                                                           \
    static constexpr auto op_ = IMPL_OP_;                                                                         \
    static constexpr auto fill_identity_op = IDENTITY_OP;                                                         \
    if (!scaled)                                                                                                  \
        return impl::partial_bitwise_binary_impl(op, op_, fill_identity_op, self, other, ARG1_NAME);              \
    return impl::scaled_bitwise_binary_impl(op, op_, fill_identity_op, self, other, ARG1_NAME);                   \
}

#define PT_DEFINE_SCALED_BINARY_OP__WITH(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, SELF_T, OTHER_T, ARG1_T, ARG1_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME##_(                                                                           \
    SELF_T self,                                                                                                        \
    OTHER_T other,                                                                                                      \
    ARG1_T ARG1_NAME,                                                                                                   \
    bool scaled) {                                                                                                      \
    static constexpr auto op_ = IMPL_OP_;                                                                               \
    static constexpr auto fill_identity_op = IDENTITY_OP;                                                               \
    static constexpr auto fill_identity_op_ = IDENTITY_OP_;                                                             \
    if (!scaled)                                                                                                        \
        return impl::partial_bitwise_binary__impl(op_, fill_identity_op, fill_identity_op_, self, other, ARG1_NAME);    \
    return impl::scaled_bitwise_binary__impl(op_, fill_identity_op, fill_identity_op_, self, other, ARG1_NAME);         \
}

#define PT_DEFINE_SCALED_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, ARG1_T, ARG1_NAME) \
PT_DEFINE_SCALED_BINARY_OP_WITH(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME) \
PT_DEFINE_SCALED_BINARY_OP_WITH(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1_T, ARG1_NAME)                          \
PT_DEFINE_SCALED_BINARY_OP_WITH(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME)                          \
PT_DEFINE_SCALED_BINARY_OP_WITH(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, const at::Tensor &, const at::Tensor &, ARG1_T, ARG1_NAME)

#define PT_DEFINE_SCALED_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH(NAME, TENSOR_OP, TENSOR_OP_, SCALAR_OP, SCALAR_OP_, IDENTITY_OP, ARG1_T, ARG1_NAME) \
PT_DEFINE_SCALED_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, ARG1_T, ARG1_NAME)                                                 \
PT_DEFINE_SCALED_BINARY_OP_WITH(NAME, SCALAR_OP, SCALAR_OP_, IDENTITY_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Scalar &, ARG1_T, ARG1_NAME)         \
PT_DEFINE_SCALED_BINARY_OP_WITH(NAME, SCALAR_OP, SCALAR_OP_, IDENTITY_OP, const at::Tensor &, const at::Scalar &, ARG1_T, ARG1_NAME)

#define PT_DEFINE_SCALED_BINARY_OPS_FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(NAME, TENSOR_OP, TENSOR_OP_, SCALAR_OP, SCALAR_OP_, IDENTITY_OP, ARG1_T, ARG1_NAME) \
PT_DEFINE_SCALED_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH(NAME, TENSOR_OP, TENSOR_OP_, SCALAR_OP, SCALAR_OP_, IDENTITY_OP, ARG1_T, ARG1_NAME)   \
PT_DEFINE_SCALED_BINARY_OP_WITH(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, const at::Scalar &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME)   \
PT_DEFINE_SCALED_BINARY_OP_WITH(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, const at::Scalar &, const at::Tensor &, ARG1_T, ARG1_NAME)

#define PT_DEFINE_SCALED_BINARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, TENSOR_OP_, IDENTITY_OP, IDENTITY_OP_, ARG1_T, ARG1_NAME) \
PT_DEFINE_SCALED_BINARY_OP__WITH(NAME, TENSOR_OP_, IDENTITY_OP, IDENTITY_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME) \
PT_DEFINE_SCALED_BINARY_OP__WITH(NAME, TENSOR_OP_, IDENTITY_OP, IDENTITY_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1_T, ARG1_NAME)                          \
PT_DEFINE_SCALED_BINARY_OP__WITH(NAME, TENSOR_OP_, IDENTITY_OP, IDENTITY_OP_, at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME)                          \
PT_DEFINE_SCALED_BINARY_OP__WITH(NAME, TENSOR_OP_, IDENTITY_OP, IDENTITY_OP_, at::Tensor &, const at::Tensor &, ARG1_T, ARG1_NAME)

#define PT_DEFINE_SCALED_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH(NAME, TENSOR_OP_, SCALAR_OP_, IDENTITY_OP, IDENTITY_OP_, ARG1_T, ARG1_NAME) \
PT_DEFINE_SCALED_BINARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, TENSOR_OP_, IDENTITY_OP, IDENTITY_OP_, ARG1_T, ARG1_NAME)                                      \
PT_DEFINE_SCALED_BINARY_OP__WITH(NAME, SCALAR_OP_, IDENTITY_OP, IDENTITY_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Scalar &, ARG1_T, ARG1_NAME)    \
PT_DEFINE_SCALED_BINARY_OP__WITH(NAME, SCALAR_OP_, IDENTITY_OP, IDENTITY_OP_, at::Tensor &, const at::Scalar &, ARG1_T, ARG1_NAME)

#define PT_DEFINE_SCALED_BINARY_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(NAME, TENSOR_OP_, SCALAR_OP_, IDENTITY_OP, IDENTITY_OP_, ARG1_T, ARG1_NAME) \
PT_DEFINE_SCALED_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH(NAME, TENSOR_OP_, SCALAR_OP_, IDENTITY_OP, IDENTITY_OP_, ARG1_T, ARG1_NAME)   \
PT_DEFINE_SCALED_BINARY_OP__WITH(NAME, TENSOR_OP_, IDENTITY_OP, IDENTITY_OP_, const at::Scalar &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME) \
PT_DEFINE_SCALED_BINARY_OP__WITH(NAME, TENSOR_OP_, IDENTITY_OP, IDENTITY_OP_, const at::Scalar &, const at::Tensor &, ARG1_T, ARG1_NAME)

#define PT_DEFINE_SCALED_BINARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, IDENTITY_OP_, ARG1_T, ARG1_NAME) \
PT_DEFINE_SCALED_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, TENSOR_OP, TENSOR_OP_, IDENTITY_OP, ARG1_T, ARG1_NAME)                                \
PT_DEFINE_SCALED_BINARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, TENSOR_OP_, IDENTITY_OP, IDENTITY_OP_, ARG1_T, ARG1_NAME)

#define PT_DEFINE_SCALED_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(NAME, TENSOR_OP, TENSOR_OP_, SCALAR_OP, SCALAR_OP_, IDENTITY_OP, IDENTITY_OP_, ARG1_T, ARG1_NAME) \
PT_DEFINE_SCALED_BINARY_OPS_FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(NAME, TENSOR_OP, TENSOR_OP_, SCALAR_OP, SCALAR_OP_, IDENTITY_OP, ARG1_T, ARG1_NAME)                                \
PT_DEFINE_SCALED_BINARY_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(NAME, TENSOR_OP_, SCALAR_OP_, IDENTITY_OP, IDENTITY_OP_, ARG1_T, ARG1_NAME)

// ~~~~~ ops registration macros ~~~~~
#define PT_BINARY_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER_T) \
utils::FunctionSchemaBuilder(#NAME).add_overload(#OVERLOAD_NAME).add_overload(#POSTFIX).arg<SELF_T>("self").arg<OTHER_T>("other").ret<TensorMaskedPair>()

#define PT_REGISTER_BINARY_OP(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER_T) \
m.def(PT_BINARY_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER_T).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, OTHER_T)>(NAME)));

#define PT_REGISTER_BINARY_OP_WITH(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
m.def(PT_BINARY_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER_T).vararg().arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, OTHER_T, ARG1_T)>(NAME)));

#define PT_REGISTER_BINARY_OP_WITH2(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER_T, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
m.def(PT_BINARY_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER_T).vararg(VARARG_INDEX).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).arg<ARG2_T>(#ARG2_NAME, #ARG2_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, OTHER_T, ARG1_T, ARG2_T)>(NAME)));

#define PT_REGISTER_BINARY_OP_WITH3(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER_T, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT) \
m.def(PT_BINARY_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER_T).vararg(VARARG_INDEX).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).arg<ARG2_T>(#ARG2_NAME, #ARG2_DEFAULT).arg<ARG3_T>(#ARG3_NAME, #ARG3_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, OTHER_T, ARG1_T, ARG2_T, ARG3_T)>(NAME)));

#define PT_REGISTER_BINARY_OP_(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER_T) \
m.def(PT_BINARY_SCHEMA_BUILDER(NAME##_, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER_T).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, OTHER_T)>(NAME##_)));

#define PT_REGISTER_BINARY_OP__WITH(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
m.def(PT_BINARY_SCHEMA_BUILDER(NAME##_, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER_T).vararg().arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, OTHER_T, ARG1_T)>(NAME##_)));

#define PT_REGISTER_BINARY_OP__WITH2(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER_T, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
m.def(PT_BINARY_SCHEMA_BUILDER(NAME##_, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER_T).vararg(VARARG_INDEX).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).arg<ARG2_T>(#ARG2_NAME, #ARG2_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, OTHER_T, ARG1_T, ARG2_T)>(NAME##_)));

#define PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX) \
PT_REGISTER_BINARY_OP(NAME, MaskedPair, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_REGISTER_BINARY_OP(NAME, MaskedPair_Tensor, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &)                   \
PT_REGISTER_BINARY_OP(NAME, Tensor_MaskedPair, POSTFIX, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>)                   \
PT_REGISTER_BINARY_OP(NAME, Tensor, POSTFIX, const at::Tensor &, const at::Tensor &)

#define PT_REGISTER_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(NAME, POSTFIX) \
PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX)                          \
PT_REGISTER_BINARY_OP(NAME, MaskedPair_Scalar, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Scalar &) \
PT_REGISTER_BINARY_OP(NAME, Tensor_Scalar, POSTFIX, const at::Tensor &, const at::Scalar &)

#define PT_REGISTER_BINARY_OPS_FORALL_TENSOR_AND_SCALAR_OVERLOADS(NAME, POSTFIX) \
PT_REGISTER_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(NAME, POSTFIX)   \
PT_REGISTER_BINARY_OP(NAME, Scalar_MaskedPair, POSTFIX, const at::Scalar &, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_REGISTER_BINARY_OP(NAME, Scalar_Tensor, POSTFIX, const at::Scalar &, const at::Tensor &)

#define PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_BINARY_OP_WITH(NAME, MaskedPair, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_BINARY_OP_WITH(NAME, MaskedPair_Tensor, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1_T, ARG1_NAME, ARG1_DEFAULT)                   \
PT_REGISTER_BINARY_OP_WITH(NAME, Tensor_MaskedPair, POSTFIX, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG1_DEFAULT)                   \
PT_REGISTER_BINARY_OP_WITH(NAME, Tensor, POSTFIX, const at::Tensor &, const at::Tensor &, ARG1_T, ARG1_NAME, ARG1_DEFAULT)

#define PT_REGISTER_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)                          \
PT_REGISTER_BINARY_OP_WITH(NAME, MaskedPair_Scalar, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Scalar &, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_BINARY_OP_WITH(NAME, Tensor_Scalar, POSTFIX, const at::Tensor &, const at::Scalar &, ARG1_T, ARG1_NAME, ARG1_DEFAULT)

#define PT_REGISTER_BINARY_OPS_FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)   \
PT_REGISTER_BINARY_OP_WITH(NAME, Scalar_MaskedPair, POSTFIX, const at::Scalar &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_BINARY_OP_WITH(NAME, Scalar_Tensor, POSTFIX, const at::Scalar &, const at::Tensor &, ARG1_T, ARG1_NAME, ARG1_DEFAULT)

#define PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_BINARY_OP_WITH2(NAME, MaskedPair, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_BINARY_OP_WITH2(NAME, MaskedPair_Tensor, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                   \
PT_REGISTER_BINARY_OP_WITH2(NAME, Tensor_MaskedPair, POSTFIX, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                   \
PT_REGISTER_BINARY_OP_WITH2(NAME, Tensor, POSTFIX, const at::Tensor &, const at::Tensor &, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

#define PT_REGISTER_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH2(NAME, POSTFIX, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                          \
PT_REGISTER_BINARY_OP_WITH2(NAME, MaskedPair_Scalar, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Scalar &, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_BINARY_OP_WITH2(NAME, Tensor_Scalar, POSTFIX, const at::Tensor &, const at::Scalar &, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

#define PT_REGISTER_BINARY_OPS_FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH2(NAME, POSTFIX, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH2(NAME, POSTFIX, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)   \
PT_REGISTER_BINARY_OP_WITH2(NAME, Scalar_MaskedPair, POSTFIX, const at::Scalar &, const_intrusive_ptr_arg_t<TensorMaskedPair>, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_BINARY_OP_WITH2(NAME, Scalar_Tensor, POSTFIX, const at::Scalar &, const at::Tensor &, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

#define PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH3(NAME, POSTFIX, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT) \
PT_REGISTER_BINARY_OP_WITH3(NAME, MaskedPair, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT) \
PT_REGISTER_BINARY_OP_WITH3(NAME, MaskedPair_Tensor, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT)                   \
PT_REGISTER_BINARY_OP_WITH3(NAME, Tensor_MaskedPair, POSTFIX, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT)                   \
PT_REGISTER_BINARY_OP_WITH3(NAME, Tensor, POSTFIX, const at::Tensor &, const at::Tensor &, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT)

#define PT_REGISTER_BINARY_OPS__FORALL_TENSOR_OVERLOADS(NAME, POSTFIX) \
PT_REGISTER_BINARY_OP_(NAME, MaskedPair, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_REGISTER_BINARY_OP_(NAME, MaskedPair_Tensor, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &)                   \
PT_REGISTER_BINARY_OP_(NAME, Tensor_MaskedPair, POSTFIX, at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>)                   \
PT_REGISTER_BINARY_OP_(NAME, Tensor, POSTFIX, at::Tensor &, const at::Tensor &)

#define PT_REGISTER_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(NAME, POSTFIX) \
PT_REGISTER_BINARY_OPS__FORALL_TENSOR_OVERLOADS(NAME, POSTFIX)                          \
PT_REGISTER_BINARY_OP_(NAME, MaskedPair_Scalar, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Scalar &) \
PT_REGISTER_BINARY_OP_(NAME, Tensor_Scalar, POSTFIX, at::Tensor &, const at::Scalar &)

#define PT_REGISTER_BINARY_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(NAME, POSTFIX) \
PT_REGISTER_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(NAME, POSTFIX)   \
PT_REGISTER_BINARY_OP_(NAME, Scalar_MaskedPair, POSTFIX, const at::Scalar &, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_REGISTER_BINARY_OP_(NAME, Scalar_Tensor, POSTFIX, const at::Scalar &, const at::Tensor &)

#define PT_REGISTER_BINARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_BINARY_OP__WITH(NAME, MaskedPair, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_BINARY_OP__WITH(NAME, MaskedPair_Tensor, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1_T, ARG1_NAME, ARG1_DEFAULT)                   \
PT_REGISTER_BINARY_OP__WITH(NAME, Tensor_MaskedPair, POSTFIX, at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG1_DEFAULT)                   \
PT_REGISTER_BINARY_OP__WITH(NAME, Tensor, POSTFIX, at::Tensor &, const at::Tensor &, ARG1_T, ARG1_NAME, ARG1_DEFAULT)

#define PT_REGISTER_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_BINARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)                          \
PT_REGISTER_BINARY_OP__WITH(NAME, MaskedPair_Scalar, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Scalar &, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_BINARY_OP__WITH(NAME, Tensor_Scalar, POSTFIX, at::Tensor &, const at::Scalar &, ARG1_T, ARG1_NAME, ARG1_DEFAULT)

#define PT_REGISTER_BINARY_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)   \
PT_REGISTER_BINARY_OP__WITH(NAME, Scalar_MaskedPair, POSTFIX, const at::Scalar &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_BINARY_OP__WITH(NAME, Scalar_Tensor, POSTFIX, const at::Scalar &, const at::Tensor &, ARG1_T, ARG1_NAME, ARG1_DEFAULT)

#define PT_REGISTER_BINARY_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_BINARY_OP__WITH2(NAME, MaskedPair, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_BINARY_OP__WITH2(NAME, MaskedPair_Tensor, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                   \
PT_REGISTER_BINARY_OP__WITH2(NAME, Tensor_MaskedPair, POSTFIX, at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                   \
PT_REGISTER_BINARY_OP__WITH2(NAME, Tensor, POSTFIX, at::Tensor &, const at::Tensor &, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

#define PT_REGISTER_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH2(NAME, POSTFIX, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_BINARY_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                          \
PT_REGISTER_BINARY_OP__WITH2(NAME, MaskedPair_Scalar, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Scalar &, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_BINARY_OP__WITH2(NAME, Tensor_Scalar, POSTFIX, at::Tensor &, const at::Scalar &, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

#define PT_REGISTER_BINARY_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH2(NAME, POSTFIX, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH2(NAME, POSTFIX, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)   \
PT_REGISTER_BINARY_OP__WITH2(NAME, Scalar_MaskedPair, POSTFIX, const at::Scalar &, const_intrusive_ptr_arg_t<TensorMaskedPair>, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_BINARY_OP__WITH2(NAME, Scalar_Tensor, POSTFIX, const at::Scalar &, const at::Tensor &, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

#define PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(NAME, POSTFIX) \
PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX)                  \
PT_REGISTER_BINARY_OPS__FORALL_TENSOR_OVERLOADS(NAME, POSTFIX)

#define PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(NAME, POSTFIX) \
PT_REGISTER_BINARY_OPS_FORALL_TENSOR_AND_SCALAR_OVERLOADS(NAME, POSTFIX)                  \
PT_REGISTER_BINARY_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(NAME, POSTFIX)

#define PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)                  \
PT_REGISTER_BINARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)

#define PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                  \
PT_REGISTER_BINARY_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

#define PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_BINARY_OPS_FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)                  \
PT_REGISTER_BINARY_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)

#define PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH2(NAME, POSTFIX, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_BINARY_OPS_FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH2(NAME, POSTFIX, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                  \
PT_REGISTER_BINARY_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH2(NAME, POSTFIX, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

namespace partialtorch {
    namespace ops {
        namespace impl {
            template<typename op_T, typename op__T, typename fill_identity_op_T,
                    typename self_T, typename other_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> partial_bitwise_binary_impl(
                    op_T &&op,
                    op__T &&op_,
                    fill_identity_op_T &&fill_identity_op,
                    const self_T &self,
                    const other_T &other,
                    Args &&... args) {
                if constexpr (std::is_same_v<c10::remove_intrusive_t<self_T>, TensorMaskedPair> ||
                              std::is_same_v<self_T, at::Tensor>) {
                    auto output_data = op.call(fill_identity_op.call(self), fill_identity_op.call(other), args...);
                    auto output_mask = utils::any_masks(utils::get_mask(self), utils::get_mask(other));
                    return masked_pair(output_data, output_mask);
                } else {
                    auto output_mask = utils::any_masks(utils::get_mask(self), utils::get_mask(other));
                    at::Tensor output_data;
                    if constexpr (!std::is_same_v<std::base_t<op__T>, nullptr_t>) {
                        output_data = op_.call(
                                at::empty_like(utils::get_data(other), at::MemoryFormat::Preserve).fill_(
                                        utils::get_data(self)),
                                fill_identity_op.call(other), args...);
                    } else {
                        output_data = op.call(
                                at::empty_like(utils::get_data(other), at::MemoryFormat::Preserve).fill_(
                                        utils::get_data(self)),
                                fill_identity_op.call(other), args...);
                    }
                    return masked_pair(output_data, output_mask);
                }
            }

            template<typename op__T, typename fill_identity_op_T, typename fill_identity_op__T,
                    typename self_T, typename other_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> partial_bitwise_binary__impl(
                    op__T &&op_,
                    fill_identity_op_T &&fill_identity_op,
                    fill_identity_op__T &&fill_identity_op_,
                    self_T &self,
                    const other_T &other,
                    Args &&... args) {
                if constexpr (std::is_same_v<c10::remove_intrusive_t<self_T>, TensorMaskedPair> ||
                              std::is_same_v<self_T, at::Tensor>) {
                    auto self_data = fill_identity_op_.call(self);
                    auto output_data = op_.call(self_data, fill_identity_op.call(other), args...);
                    auto output_mask = utils::any_masks_(utils::get_mask(self), utils::get_mask(other));
                    if constexpr (std::is_same_v<self_T, c10::intrusive_ptr<TensorMaskedPair>>) {
                        return self;
                    } else {
                        return masked_pair(output_data, output_mask);
                    }
                } else {
                    auto output_mask = utils::any_masks(utils::get_mask(self), utils::get_mask(other));
                    auto output_data = op_.call(
                            at::empty_like(utils::get_data(other), at::MemoryFormat::Preserve).fill_(
                                    utils::get_data(self)),
                            fill_identity_op.call(other), args...);
                    return masked_pair(output_data, output_mask);
                }
            }

            template<typename op_T, typename op__T, typename fill_identity_op_T,
                    typename self_T, typename other_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> scaled_bitwise_binary_impl(
                    op_T &&op,
                    op__T &&op_,
                    fill_identity_op_T &&fill_identity_op,
                    const self_T &self,
                    const other_T &other,
                    Args &&... args) {
                if constexpr (std::is_same_v<c10::remove_intrusive_t<self_T>, TensorMaskedPair> ||
                              std::is_same_v<self_T, at::Tensor>) {
                    auto output_data = op.call(fill_identity_op.call(self),
                                               fill_identity_op.call(other), args...);
                    auto mask_ratio_options = output_data.options();
                    at::Tensor mask_ratio;
                    {
                        at::NoGradGuard g;
                        mask_ratio = at::add(
                                utils::get_tensor_mask(self, mask_ratio_options),
                                utils::get_tensor_mask(other, mask_ratio_options));
                        utils::izero_ldiv_(mask_ratio, 2);
                    }
                    output_data.mul_(mask_ratio);
                    auto output_mask = utils::any_masks(utils::get_mask(self), utils::get_mask(other));
                    return masked_pair(output_data, output_mask);
                } else {
                    at::Tensor output_data;
                    if constexpr (!std::is_same_v<std::base_t<op__T>, nullptr_t>) {
                        output_data = op_.call(
                                at::empty_like(utils::get_data(other), at::MemoryFormat::Preserve).fill_(
                                        utils::get_data(self)),
                                fill_identity_op.call(other), args...);
                    } else {
                        output_data = op.call(
                                at::empty_like(utils::get_data(other), at::MemoryFormat::Preserve).fill_(
                                        utils::get_data(self)),
                                fill_identity_op.call(other), args...);
                    }
                    auto mask_ratio_options = output_data.options();
                    at::Tensor mask_ratio;
                    {
                        at::NoGradGuard g;
                        mask_ratio = at::add(
                                at::ones_like(utils::get_data(other), mask_ratio_options),
                                utils::get_tensor_mask(other, mask_ratio_options));
                        utils::izero_ldiv_(mask_ratio, 2);
                    }
                    output_data.mul_(mask_ratio);
                    auto output_mask = utils::any_masks(utils::get_mask(self), utils::get_mask(other));
                    return masked_pair(output_data, output_mask);
                }
            }

            template<typename op__T, typename fill_identity_op_T, typename fill_identity_op__T,
                    typename self_T, typename other_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> scaled_bitwise_binary__impl(
                    op__T &&op_,
                    fill_identity_op_T &&fill_identity_op,
                    fill_identity_op__T &&fill_identity_op_,
                    self_T &self,
                    const other_T &other,
                    Args &&... args) {
                if constexpr (std::is_same_v<c10::remove_intrusive_t<self_T>, TensorMaskedPair> ||
                              std::is_same_v<self_T, at::Tensor>) {
                    auto self_data = utils::get_data(self);
                    auto output_data = op_.call(self_data, fill_identity_op.call(other), args...);
                    auto mask_ratio_options = output_data.options();
                    at::Tensor mask_ratio;
                    {
                        at::NoGradGuard g;
                        mask_ratio = at::add(
                                utils::get_tensor_mask(self, mask_ratio_options),
                                utils::get_tensor_mask(other, mask_ratio_options));
                        utils::izero_ldiv_(mask_ratio, 2);
                    }
                    output_data.mul_(mask_ratio);
                    auto output_mask = utils::any_masks(utils::get_mask(self), utils::get_mask(other));
                    if constexpr (std::is_same_v<self_T, c10::intrusive_ptr<TensorMaskedPair>>) {
                        return self;
                    } else {
                        return masked_pair(output_data, output_mask);
                    }
                } else {
                    auto output_data = op_.call(
                            at::empty_like(utils::get_data(other), at::MemoryFormat::Preserve).fill_(
                                    utils::get_data(self)),
                            fill_identity_op.call(other), args...);
                    auto mask_ratio_options = output_data.options();
                    at::Tensor mask_ratio;
                    {
                        at::NoGradGuard g;
                        mask_ratio = at::add(
                                at::ones_like(utils::get_data(other), mask_ratio_options),
                                utils::get_tensor_mask(other, mask_ratio_options));
                        utils::izero_ldiv_(mask_ratio, 2);
                    }
                    output_data.mul_(mask_ratio);
                    auto output_mask = utils::any_masks(utils::get_mask(self), utils::get_mask(other));
                    return masked_pair(output_data, output_mask);
                }
            }
        }

        // logical
        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                partial_bitwise_and, at::_ops::bitwise_and_Tensor(), at::_ops::bitwise_and__Tensor(),
                at::_ops::bitwise_and_Scalar(), at::_ops::bitwise_and__Scalar(),
                utils::_ops::fill_identity_zeros(), utils::_ops::fill_identity_zeros_())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                partial_bitwise_or, at::_ops::bitwise_or_Tensor(), at::_ops::bitwise_or__Tensor(),
                at::_ops::bitwise_or_Scalar(), at::_ops::bitwise_or__Scalar(),
                utils::_ops::fill_identity_ones(), utils::_ops::fill_identity_ones_())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                partial_bitwise_xor, at::_ops::bitwise_xor_Tensor(), at::_ops::bitwise_xor__Tensor(),
                at::_ops::bitwise_xor_Scalar(), at::_ops::bitwise_xor__Scalar(),
                utils::_ops::fill_identity_zeros(), utils::_ops::fill_identity_zeros_())

        // arithmetics
        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(
                partial_add, at::_ops::add_Tensor(), at::_ops::add__Tensor(),
                at::_ops::add_Scalar(), at::_ops::add__Scalar(),
                utils::_ops::fill_identity_zeros(), utils::_ops::fill_identity_zeros_(),
                const at::Scalar &, alpha)

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(
                partial_sub, at::_ops::sub_Tensor(), at::_ops::sub__Tensor(),
                at::_ops::sub_Scalar(), at::_ops::sub__Scalar(),
                utils::_ops::fill_identity_zeros(), utils::_ops::fill_identity_zeros_(),
                const at::Scalar &, alpha)

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(
                partial_subtract, at::_ops::subtract_Tensor(), at::_ops::subtract__Tensor(),
                at::_ops::subtract_Scalar(), at::_ops::subtract__Scalar(),
                utils::_ops::fill_identity_zeros(), utils::_ops::fill_identity_zeros_(),
                const at::Scalar &, alpha)

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                partial_mul, at::_ops::mul_Tensor(), at::_ops::mul__Tensor(),
                at::_ops::mul_Scalar(), at::_ops::mul__Scalar(),
                utils::_ops::fill_identity_ones(), utils::_ops::fill_identity_ones_())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                partial_multiply, at::_ops::multiply_Tensor(), at::_ops::multiply__Tensor(),
                at::_ops::multiply_Scalar(), at::_ops::multiply__Scalar(),
                utils::_ops::fill_identity_ones(), utils::_ops::fill_identity_ones_())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                partial_div, at::_ops::div_Tensor(), at::_ops::div__Tensor(),
                at::_ops::div_Scalar(), at::_ops::div__Scalar(),
                utils::_ops::fill_identity_ones(), utils::_ops::fill_identity_ones_())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(
                partial_div, at::_ops::div_Tensor_mode(), at::_ops::div__Tensor_mode(),
                at::_ops::div_Scalar_mode(), at::_ops::div__Scalar_mode(),
                utils::_ops::fill_identity_ones(), utils::_ops::fill_identity_ones_(),
                c10::optional<c10::string_view>, rounding_mode)

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                partial_divide, at::_ops::divide_Tensor(), at::_ops::divide__Tensor(),
                at::_ops::divide_Scalar(), at::_ops::divide__Scalar(),
                utils::_ops::fill_identity_ones(), utils::_ops::fill_identity_ones_())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(
                partial_divide, at::_ops::divide_Tensor_mode(), at::_ops::divide__Tensor_mode(),
                at::_ops::divide_Scalar_mode(), at::_ops::divide__Scalar_mode(),
                utils::_ops::fill_identity_ones(), utils::_ops::fill_identity_ones_(),
                c10::optional<c10::string_view>, rounding_mode)

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                partial_floor_divide, at::_ops::floor_divide(), at::_ops::floor_divide__Tensor(),
                at::_ops::floor_divide_Scalar(), at::_ops::floor_divide__Scalar(),
                utils::_ops::fill_identity_ones(), utils::_ops::fill_identity_ones_())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                partial_true_divide, at::_ops::true_divide_Tensor(), at::_ops::true_divide__Tensor(),
                at::_ops::true_divide_Scalar(), at::_ops::true_divide__Scalar(),
                utils::_ops::fill_identity_ones(), utils::_ops::fill_identity_ones_())

        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                partial_logaddexp, at::_ops::logaddexp(), nullptr,
                utils::_ops::fill_identity_neginf())

        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                partial_logaddexp2, at::_ops::logaddexp2(), nullptr,
                utils::_ops::fill_identity_neginf())

        // scaled arithmetics
        PT_DEFINE_SCALED_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(
                partial_add, at::_ops::add_Tensor(), at::_ops::add__Tensor(),
                at::_ops::add_Scalar(), at::_ops::add__Scalar(),
                utils::_ops::fill_identity_zeros(), utils::_ops::fill_identity_zeros_(),
                const at::Scalar &, alpha)

        PT_DEFINE_SCALED_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(
                partial_sub, at::_ops::sub_Tensor(), at::_ops::sub__Tensor(),
                at::_ops::sub_Scalar(), at::_ops::sub__Scalar(),
                utils::_ops::fill_identity_zeros(), utils::_ops::fill_identity_zeros_(),
                const at::Scalar &, alpha)

        PT_DEFINE_SCALED_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(
                partial_subtract, at::_ops::subtract_Tensor(), at::_ops::subtract__Tensor(),
                at::_ops::subtract_Scalar(), at::_ops::subtract__Scalar(),
                utils::_ops::fill_identity_zeros(), utils::_ops::fill_identity_zeros_(),
                const at::Scalar &, alpha)

        // comparison
        namespace impl {
            struct isclose_no_equal_nan {
                static at::Tensor call(const at::Tensor &self, const at::Tensor &other, double rtol, double atol) {
                    return at::_ops::isclose::call(self, other, rtol, atol, false);
                }
            };

            template<typename self_T, typename other_T>
            static C10_ALWAYS_INLINE bool partial_equal_impl(
                    const self_T &self,
                    const other_T &other) {
                auto both_mask = utils::all_masks(utils::get_mask(self), utils::get_mask(other));
                if (both_mask.has_value())
                    return at::_ops::equal::call(utils::get_data(self).index({both_mask.value()}),
                                                 utils::get_data(other).index({both_mask.value()}));
                return at::_ops::equal::call(utils::get_data(self), utils::get_data(other));
            }
        }

        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                partial_isclose, impl::isclose_no_equal_nan(), nullptr,
                utils::_ops::fill_identity_nan(),
                double, rtol, double, atol)

        bool partial_equal(const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                           const_intrusive_ptr_arg_t<TensorMaskedPair> other) {
            return impl::partial_equal_impl(self, other);
        }

        bool partial_equal(const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                           const at::Tensor &other) {
            return impl::partial_equal_impl(self, other);
        }

        bool partial_equal(const at::Tensor &self,
                           const_intrusive_ptr_arg_t<TensorMaskedPair> other) {
            return impl::partial_equal_impl(self, other);
        }

        bool partial_equal(const at::Tensor &self,
                           const at::Tensor &other) {
            return at::_ops::equal::call(self, other);
        }

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                partial_eq, at::_ops::eq_Tensor(), at::_ops::eq__Tensor(),
                at::_ops::eq_Scalar(), at::_ops::eq__Scalar(),
                utils::_ops::fill_identity_nan(), utils::_ops::fill_identity_nan_())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                partial_ne, at::_ops::ne_Tensor(), at::_ops::ne__Tensor(),
                at::_ops::ne_Scalar(), at::_ops::ne__Scalar(),
                utils::_ops::fill_identity_nan(), utils::_ops::fill_identity_nan_())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                partial_not_equal, at::_ops::not_equal_Tensor(), at::_ops::not_equal__Tensor(),
                at::_ops::not_equal_Scalar(), at::_ops::not_equal__Scalar(),
                utils::_ops::fill_identity_nan(), utils::_ops::fill_identity_nan_())

        // min max
        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                partial_min, at::_ops::min_other(), nullptr,
                utils::_ops::fill_identity_posinf())

        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                partial_max, at::_ops::max_other(), nullptr,
                utils::_ops::fill_identity_neginf())

        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                partial_minimum, at::_ops::minimum(), nullptr,
                utils::_ops::fill_identity_posinf())

        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                partial_maxium, at::_ops::maximum(), nullptr,
                utils::_ops::fill_identity_neginf())

        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                partial_fmin, at::_ops::fmin(), nullptr,
                utils::_ops::fill_identity_nan())

        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                partial_fmax, at::_ops::fmax(), nullptr,
                utils::_ops::fill_identity_nan())

        TORCH_LIBRARY_FRAGMENT(partialtorch, m) {
            // logical
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(partial_bitwise_and,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(partial_bitwise_or,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(partial_bitwise_xor,)

            // arithmetics
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(
                    partial_add, ,
                    const at::Scalar &, alpha, 1)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(
                    partial_sub, ,
                    const at::Scalar &, alpha, 1)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(
                    partial_subtract, ,
                    const at::Scalar &, alpha, 1)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(partial_mul,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(partial_multiply,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(partial_div,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(
                    partial_div, mode,
                    c10::optional<c10::string_view>, rounding_mode, None)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(partial_divide,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(
                    partial_divide, mode,
                    c10::optional<c10::string_view>, rounding_mode, None)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(partial_floor_divide,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(partial_true_divide,)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS(partial_logaddexp,)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS(partial_logaddexp2,)

            // scaled arithmetics
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH2(
                    partial_add, scaled, ,
                    const at::Scalar &, alpha, 1,
                    bool, scaled,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH2(
                    partial_sub, scaled, ,
                    const at::Scalar &, alpha, 1,
                    bool, scaled,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH2(
                    partial_subtract, scaled, ,
                    const at::Scalar &, alpha, 1,
                    bool, scaled,)

            // comparison
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    partial_isclose, , -1,
                    double, rtol, 1e-05,
                    double, atol, 1e-08)

            m.def(utils::FunctionSchemaBuilder("partial_equal")
                          .overload("MaskedPair")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("other").ret<bool>().schema().c_str(),
                  TORCH_FN(static_cast<bool (*)(const_intrusive_ptr_arg_t<TensorMaskedPair>,
                                                const_intrusive_ptr_arg_t<TensorMaskedPair>)>(partial_equal)));
            m.def(utils::FunctionSchemaBuilder("partial_equal")
                          .overload("MaskedPair_Tensor")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .arg<const at::Tensor &>("other").ret<bool>().schema().c_str(),
                  TORCH_FN(static_cast<bool (*)(const_intrusive_ptr_arg_t<TensorMaskedPair>,
                                                const at::Tensor &)>(partial_equal)));
            m.def(utils::FunctionSchemaBuilder("partial_equal")
                          .overload("Tensor_MaskedPair")
                          .arg<const at::Tensor &>("self")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("other").ret<bool>().schema().c_str(),
                  TORCH_FN(static_cast<bool (*)(const at::Tensor &,
                                                const_intrusive_ptr_arg_t<TensorMaskedPair>)>(partial_equal)));
            m.def(utils::FunctionSchemaBuilder("partial_equal")
                          .overload("Tensor")
                          .arg<const at::Tensor &>("self")
                          .arg<const at::Tensor &>("other").ret<bool>().schema().c_str(),
                  TORCH_FN(static_cast<bool (*)(const at::Tensor &,
                                                const at::Tensor &)>(partial_equal)));

            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(partial_eq,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(partial_ne,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(partial_not_equal,)

            // min max
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS(partial_min, other)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS(partial_max, other)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS(partial_minimum,)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS(partial_maxium,)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS(partial_fmin,)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS(partial_fmax,)
        }
    }
}
