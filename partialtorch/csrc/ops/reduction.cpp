#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <torch/library.h>

#include "../MaskedPair.h"
#include "izero_div.h"
#include "utils/mask_utils.h"
#include "utils/reduction_utils.h"
#include "utils/result_with_indices.h"
#include "utils/schema_utils.h"
#include "utils/torch_library_utils.h"

// ~~~~~ ops definition macros ~~~~~
#define PT_DEFINE_REDUCTION_OP(NAME, IMPL_OP, IDENTITY_OP, REDUCE, SELF_T) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                 \
    SELF_T self) {                                                         \
    static constexpr auto op = IMPL_OP;                                    \
    static constexpr auto fill_identity_op = IDENTITY_OP;                  \
    return impl::reduction_impl<REDUCE>(op, fill_identity_op, self);       \
}

#define PT_DEFINE_REDUCTION_OP_WITH(NAME, IMPL_OP, IDENTITY_OP, REDUCE, SELF_T, ARG1_T, ARG1_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                         \
    SELF_T self,                                                                                   \
    ARG1_T ARG1_NAME) {                                                                            \
    static constexpr auto op = IMPL_OP;                                                            \
    static constexpr auto fill_identity_op = IDENTITY_OP;                                          \
    return impl::reduction_impl<REDUCE>(op, fill_identity_op, self, ARG1_NAME);                    \
}

#define PT_DEFINE_REDUCTION_DIM_OP(NAME, IMPL_OP, IDENTITY_OP, REDUCE, SELF_T, DIM_T) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                            \
    SELF_T self,                                                                      \
    DIM_T dim) {                                                                      \
    static constexpr auto op = IMPL_OP;                                               \
    static constexpr auto fill_identity_op = IDENTITY_OP;                             \
    return impl::reduction_dim_impl<REDUCE>(op, fill_identity_op, self, dim);         \
}

#define PT_DEFINE_REDUCTION_DIM_OP_WITH(NAME, IMPL_OP, IDENTITY_OP, REDUCE, SELF_T, DIM_T, ARG1_T, ARG1_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                                    \
    SELF_T self,                                                                                              \
    DIM_T dim,                                                                                                \
    ARG1_T ARG1_NAME) {                                                                                       \
    static constexpr auto op = IMPL_OP;                                                                       \
    static constexpr auto fill_identity_op = IDENTITY_OP;                                                     \
    return impl::reduction_dim_impl<REDUCE>(op, fill_identity_op, self, dim, ARG1_NAME);                      \
}

#define PT_DEFINE_REDUCTION_KEEPDIM_OP(NAME, IMPL_OP, IDENTITY_OP, REDUCE, SELF_T, DIM_T) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                \
    SELF_T self,                                                                          \
    DIM_T dim,                                                                            \
    bool keepdim) {                                                                       \
    static constexpr auto op = IMPL_OP;                                                   \
    static constexpr auto fill_identity_op = IDENTITY_OP;                                 \
    return impl::reduction_dim_impl<REDUCE>(op, fill_identity_op, self, dim, keepdim);    \
}

#define PT_DEFINE_REDUCTION_KEEPDIM_OP_WITH(NAME, IMPL_OP, IDENTITY_OP, REDUCE, SELF_T, DIM_T, ARG1_T, ARG1_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                                        \
    SELF_T self,                                                                                                  \
    DIM_T dim,                                                                                                    \
    bool keepdim,                                                                                                 \
    ARG1_T ARG1_NAME) {                                                                                           \
    static constexpr auto op = IMPL_OP;                                                                           \
    static constexpr auto fill_identity_op = IDENTITY_OP;                                                         \
    return impl::reduction_dim_impl<REDUCE>(op, fill_identity_op, self, dim, keepdim, ARG1_NAME);                 \
}

#define PT_DEFINE_REDUCTION_DIM_OP_(NAME, IMPL_OP_, IDENTITY_OP_, SELF_T, DIM_T) \
c10::intrusive_ptr<TensorMaskedPair> NAME##_(                                    \
    SELF_T self,                                                                 \
    DIM_T dim) {                                                                 \
    static constexpr auto op_ = IMPL_OP_;                                        \
    static constexpr auto fill_identity_op_ = IDENTITY_OP_;                      \
    return impl::reduction__dim_impl(op_, fill_identity_op_, self, dim);         \
}

#define PT_DEFINE_REDUCTION_DIM_OP__WITH(NAME, IMPL_OP_, IDENTITY_OP_, SELF_T, DIM_T, ARG1_T, ARG1_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME##_(                                                            \
    SELF_T self,                                                                                         \
    DIM_T dim,                                                                                           \
    ARG1_T ARG1_NAME) {                                                                                  \
    static constexpr auto op_ = IMPL_OP_;                                                                \
    static constexpr auto fill_identity_op_ = IDENTITY_OP_;                                              \
    return impl::reduction__dim_impl(op_, fill_identity_op_, self, dim, ARG1_NAME);                      \
}

#define PT_DEFINE_REDUCTION_INDICES_DIM_OP(NAME, IMPL_OP, IDENTITY_OP, REDUCE, SELF_T, DIM_T) \
std::tuple<c10::intrusive_ptr<TensorMaskedPair>, at::Tensor> NAME(                            \
    SELF_T self,                                                                              \
    DIM_T dim) {                                                                              \
    static constexpr auto op = IMPL_OP;                                                       \
    static constexpr auto fill_identity_op = IDENTITY_OP;                                     \
    return impl::reduction_indices_dim_impl<REDUCE>(op, fill_identity_op, self, dim);         \
}

#define PT_DEFINE_REDUCTION_INDICES_KEEPDIM_OP(NAME, IMPL_OP, IDENTITY_OP, REDUCE, SELF_T, DIM_T) \
std::tuple<c10::intrusive_ptr<TensorMaskedPair>, at::Tensor> NAME(                                \
    SELF_T self,                                                                                  \
    DIM_T dim,                                                                                    \
    bool keepdim) {                                                                               \
    static constexpr auto op = IMPL_OP;                                                           \
    static constexpr auto fill_identity_op = IDENTITY_OP;                                         \
    return impl::reduction_indices_dim_impl<REDUCE>(op, fill_identity_op, self, dim, keepdim);    \
}

#define PT_DEFINE_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, IDENTITY_OP, REDUCE) \
PT_DEFINE_REDUCTION_OP(NAME, IMPL_OP, IDENTITY_OP, REDUCE, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DEFINE_REDUCTION_OP(NAME, IMPL_OP, IDENTITY_OP, REDUCE, const at::Tensor &)

#define PT_DEFINE_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP, IDENTITY_OP, REDUCE, ARG1_T, ARG1_NAME) \
PT_DEFINE_REDUCTION_OP_WITH(NAME, IMPL_OP, IDENTITY_OP, REDUCE, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME) \
PT_DEFINE_REDUCTION_OP_WITH(NAME, IMPL_OP, IDENTITY_OP, REDUCE, const at::Tensor &, ARG1_T, ARG1_NAME)

#define PT_DEFINE_REDUCTION_DIM_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, IDENTITY_OP, REDUCE, DIM_T) \
PT_DEFINE_REDUCTION_DIM_OP(NAME, IMPL_OP, IDENTITY_OP, REDUCE, const_intrusive_ptr_arg_t<TensorMaskedPair>, DIM_T) \
PT_DEFINE_REDUCTION_DIM_OP(NAME, IMPL_OP, IDENTITY_OP, REDUCE, const at::Tensor &, DIM_T)

#define PT_DEFINE_REDUCTION_DIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP, IDENTITY_OP, REDUCE, DIM_T, ARG1_T, ARG1_NAME) \
PT_DEFINE_REDUCTION_DIM_OP_WITH(NAME, IMPL_OP, IDENTITY_OP, REDUCE, const_intrusive_ptr_arg_t<TensorMaskedPair>, DIM_T, ARG1_T, ARG1_NAME) \
PT_DEFINE_REDUCTION_DIM_OP_WITH(NAME, IMPL_OP, IDENTITY_OP, REDUCE, const at::Tensor &, DIM_T, ARG1_T, ARG1_NAME)

#define PT_DEFINE_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, IDENTITY_OP, REDUCE, DIM_T) \
PT_DEFINE_REDUCTION_KEEPDIM_OP(NAME, IMPL_OP, IDENTITY_OP, REDUCE, const_intrusive_ptr_arg_t<TensorMaskedPair>, DIM_T) \
PT_DEFINE_REDUCTION_KEEPDIM_OP(NAME, IMPL_OP, IDENTITY_OP, REDUCE, const at::Tensor &, DIM_T)

#define PT_DEFINE_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP, IDENTITY_OP, REDUCE, DIM_T, ARG1_T, ARG1_NAME) \
PT_DEFINE_REDUCTION_KEEPDIM_OP_WITH(NAME, IMPL_OP, IDENTITY_OP, REDUCE, const_intrusive_ptr_arg_t<TensorMaskedPair>, DIM_T, ARG1_T, ARG1_NAME) \
PT_DEFINE_REDUCTION_KEEPDIM_OP_WITH(NAME, IMPL_OP, IDENTITY_OP, REDUCE, const at::Tensor &, DIM_T, ARG1_T, ARG1_NAME)

#define PT_DEFINE_REDUCTION_INDICES_DIM_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, IDENTITY_OP, REDUCE, DIM_T) \
PT_DEFINE_REDUCTION_INDICES_DIM_OP(NAME, IMPL_OP, IDENTITY_OP, REDUCE, const_intrusive_ptr_arg_t<TensorMaskedPair>, DIM_T) \
PT_DEFINE_REDUCTION_INDICES_DIM_OP(NAME, IMPL_OP, IDENTITY_OP, REDUCE, const at::Tensor &, DIM_T)

#define PT_DEFINE_REDUCTION_INDICES_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, IDENTITY_OP, REDUCE, DIM_T) \
PT_DEFINE_REDUCTION_INDICES_KEEPDIM_OP(NAME, IMPL_OP, IDENTITY_OP, REDUCE, const_intrusive_ptr_arg_t<TensorMaskedPair>, DIM_T) \
PT_DEFINE_REDUCTION_INDICES_KEEPDIM_OP(NAME, IMPL_OP, IDENTITY_OP, REDUCE, const at::Tensor &, DIM_T)

#define PT_DEFINE_REDUCTION_DIM_OPS__FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP_, IDENTITY_OP_, DIM_T) \
PT_DEFINE_REDUCTION_DIM_OP_(NAME, IMPL_OP_, IDENTITY_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, DIM_T) \
PT_DEFINE_REDUCTION_DIM_OP_(NAME, IMPL_OP_, IDENTITY_OP_, at::Tensor &, DIM_T)

#define PT_DEFINE_REDUCTION_DIM_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP_, IDENTITY_OP_, DIM_T, ARG1_T, ARG1_NAME) \
PT_DEFINE_REDUCTION_DIM_OP__WITH(NAME, IMPL_OP_, IDENTITY_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, DIM_T, ARG1_T, ARG1_NAME) \
PT_DEFINE_REDUCTION_DIM_OP__WITH(NAME, IMPL_OP_, IDENTITY_OP_, at::Tensor &, DIM_T, ARG1_T, ARG1_NAME)

#define PT_DEFINE_REDUCTION_AND_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, KEEPDIM_IMPL_OP, IDENTITY_OP, REDUCE, DIM_T) \
PT_DEFINE_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, IDENTITY_OP, REDUCE)                                             \
PT_DEFINE_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(NAME, KEEPDIM_IMPL_OP, IDENTITY_OP, REDUCE, DIM_T)

#define PT_DEFINE_REDUCTION_AND_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP, KEEPDIM_IMPL_OP, IDENTITY_OP, REDUCE, DIM_T, ARG1_T, ARG1_NAME) \
PT_DEFINE_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP, IDENTITY_OP, REDUCE, ARG1_T, ARG1_NAME)                                             \
PT_DEFINE_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, KEEPDIM_IMPL_OP, IDENTITY_OP, REDUCE, DIM_T, ARG1_T, ARG1_NAME)

#define PT_DEFINE_REDUCTION_DIM_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, IDENTITY_OP, IMPL_OP_, IDENTITY_OP_, DIM_T) \
PT_DEFINE_REDUCTION_DIM_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, IDENTITY_OP, false, DIM_T)                                   \
PT_DEFINE_REDUCTION_DIM_OPS__FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP_, IDENTITY_OP_, DIM_T)

#define PT_DEFINE_REDUCTION_DIM_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP, IDENTITY_OP, IMPL_OP_, IDENTITY_OP_, DIM_T, ARG1_T, ARG1_NAME) \
PT_DEFINE_REDUCTION_DIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP, IDENTITY_OP, false, DIM_T, ARG1_T, ARG1_NAME)                                   \
PT_DEFINE_REDUCTION_DIM_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP_, IDENTITY_OP_, DIM_T, ARG1_T, ARG1_NAME)

// scaled reduction
#define PT_DEFINE_SCALED_REDUCTION_OP(NAME, IMPL_OP, IDENTITY_OP, CANONICAL_SCALE, SELF_T) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                 \
    SELF_T self,                                                                           \
    bool scaled) {                                                                         \
    static constexpr auto op = IMPL_OP;                                                    \
    static constexpr auto fill_identity_op = IDENTITY_OP;                                  \
    static constexpr auto canonical_scale = CANONICAL_SCALE;                               \
    if (!scaled)                                                                           \
        return impl::reduction_impl(op, fill_identity_op, self);                           \
    return impl::scaled_reduction_impl(op, fill_identity_op, canonical_scale, self);       \
}

#define PT_DEFINE_SCALED_REDUCTION_OP_WITH(NAME, IMPL_OP, IDENTITY_OP, CANONICAL_SCALE, SELF_T, ARG1_T, ARG1_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                                         \
    SELF_T self,                                                                                                   \
    ARG1_T ARG1_NAME,                                                                                              \
    bool scaled) {                                                                                                 \
    static constexpr auto op = IMPL_OP;                                                                            \
    static constexpr auto fill_identity_op = IDENTITY_OP;                                                          \
    static constexpr auto canonical_scale = CANONICAL_SCALE;                                                       \
    if (!scaled)                                                                                                   \
        return impl::reduction_impl(op, fill_identity_op, self, ARG1_NAME);                                        \
    return impl::scaled_reduction_impl(op, fill_identity_op, canonical_scale, self, ARG1_NAME);                    \
}

#define PT_DEFINE_SCALED_REDUCTION_KEEPDIM_OP(NAME, IMPL_OP, IDENTITY_OP, CANONICAL_SCALE, SELF_T, DIM_T) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                                \
    SELF_T self,                                                                                          \
    DIM_T dim,                                                                                            \
    bool keepdim,                                                                                         \
    bool scaled) {                                                                                        \
    static constexpr auto op = IMPL_OP;                                                                   \
    static constexpr auto fill_identity_op = IDENTITY_OP;                                                 \
    static constexpr auto canonical_scale = CANONICAL_SCALE;                                              \
    if (!scaled)                                                                                          \
        return impl::reduction_dim_impl(op, fill_identity_op, self, dim, keepdim);                        \
    return impl::scaled_reduction_dim_impl(op, fill_identity_op, canonical_scale, self, dim, keepdim);    \
}

#define PT_DEFINE_SCALED_REDUCTION_KEEPDIM_OP_WITH(NAME, IMPL_OP, IDENTITY_OP, CANONICAL_SCALE, SELF_T, DIM_T, ARG1_T, ARG1_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                                                        \
    SELF_T self,                                                                                                                  \
    DIM_T dim,                                                                                                                    \
    bool keepdim,                                                                                                                 \
    ARG1_T ARG1_NAME,                                                                                                             \
    bool scaled) {                                                                                                                \
    static constexpr auto op = IMPL_OP;                                                                                           \
    static constexpr auto fill_identity_op = IDENTITY_OP;                                                                         \
    static constexpr auto canonical_scale = CANONICAL_SCALE;                                                                      \
    if (!scaled)                                                                                                                  \
        return impl::reduction_dim_impl(op, fill_identity_op, self, dim, keepdim, ARG1_NAME);                                     \
    return impl::scaled_reduction_dim_impl(op, fill_identity_op, canonical_scale, self, dim, keepdim, ARG1_NAME);                 \
}

#define PT_DEFINE_SCALED_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, IDENTITY_OP, CANONICAL_SCALE) \
PT_DEFINE_SCALED_REDUCTION_OP(NAME, IMPL_OP, IDENTITY_OP, CANONICAL_SCALE, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DEFINE_SCALED_REDUCTION_OP(NAME, IMPL_OP, IDENTITY_OP, CANONICAL_SCALE, const at::Tensor &)

#define PT_DEFINE_SCALED_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP, IDENTITY_OP, CANONICAL_SCALE, ARG1_T, ARG1_NAME) \
PT_DEFINE_SCALED_REDUCTION_OP_WITH(NAME, IMPL_OP, IDENTITY_OP, CANONICAL_SCALE, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME) \
PT_DEFINE_SCALED_REDUCTION_OP_WITH(NAME, IMPL_OP, IDENTITY_OP, CANONICAL_SCALE, const at::Tensor &, ARG1_T, ARG1_NAME)

#define PT_DEFINE_SCALED_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, IDENTITY_OP, CANONICAL_SCALE, DIM_T) \
PT_DEFINE_SCALED_REDUCTION_KEEPDIM_OP(NAME, IMPL_OP, IDENTITY_OP, CANONICAL_SCALE, const_intrusive_ptr_arg_t<TensorMaskedPair>, DIM_T) \
PT_DEFINE_SCALED_REDUCTION_KEEPDIM_OP(NAME, IMPL_OP, IDENTITY_OP, CANONICAL_SCALE, const at::Tensor &, DIM_T)

#define PT_DEFINE_SCALED_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP, IDENTITY_OP, CANONICAL_SCALE, DIM_T, ARG1_T, ARG1_NAME) \
PT_DEFINE_SCALED_REDUCTION_KEEPDIM_OP_WITH(NAME, IMPL_OP, IDENTITY_OP, CANONICAL_SCALE, const_intrusive_ptr_arg_t<TensorMaskedPair>, DIM_T, ARG1_T, ARG1_NAME) \
PT_DEFINE_SCALED_REDUCTION_KEEPDIM_OP_WITH(NAME, IMPL_OP, IDENTITY_OP, CANONICAL_SCALE, const at::Tensor &, DIM_T, ARG1_T, ARG1_NAME)

#define PT_DEFINE_SCALED_REDUCTION_AND_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, KEEPDIM_IMPL_OP, IDENTITY_OP, CANONICAL_SCALE, DIM_T) \
PT_DEFINE_SCALED_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, IDENTITY_OP, CANONICAL_SCALE)                                             \
PT_DEFINE_SCALED_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(NAME, KEEPDIM_IMPL_OP, IDENTITY_OP, CANONICAL_SCALE, DIM_T)

#define PT_DEFINE_SCALED_REDUCTION_AND_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP, KEEPDIM_IMPL_OP, IDENTITY_OP, CANONICAL_SCALE, DIM_T, ARG1_T, ARG1_NAME) \
PT_DEFINE_SCALED_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP, IDENTITY_OP, CANONICAL_SCALE, ARG1_T, ARG1_NAME)                                             \
PT_DEFINE_SCALED_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, KEEPDIM_IMPL_OP, IDENTITY_OP, CANONICAL_SCALE, DIM_T, ARG1_T, ARG1_NAME)

// ~~~~~ ops registration macros ~~~~~
#define PT_REDUCTION_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T) \
utils::FunctionSchemaBuilder(#NAME).add_overload(#OVERLOAD_NAME).add_overload(#POSTFIX).arg<SELF_T>("self").ret<TensorMaskedPair>()

#define PT_MINMAX_DIM_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T) \
PT_REDUCTION_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).ret<at::Tensor>()

#define PT_REGISTER_REDUCTION_OP(NAME, OVERLOAD_NAME, POSTFIX, SELF_T) \
m.def(PT_REDUCTION_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T)>(NAME)));

#define PT_REGISTER_REDUCTION_OP_WITH(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
m.def(PT_REDUCTION_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).vararg().arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, ARG1_T)>(NAME)));

#define PT_REGISTER_REDUCTION_OP_WITH2(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
m.def(PT_REDUCTION_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).vararg().arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).arg<ARG2_T>(#ARG2_NAME, #ARG2_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, ARG1_T, ARG2_T)>(NAME)));

#define PT_REGISTER_REDUCTION_DIM_OP(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, DIM_T, DIM_DEFAULT) \
m.def(PT_REDUCTION_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).arg<DIM_T>("dim", #DIM_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, int64_t)>(NAME)));

#define PT_REGISTER_REDUCTION_DIM_OP_WITH(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, DIM_T, DIM_DEFAULT, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
m.def(PT_REDUCTION_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).arg<DIM_T>("dim", #DIM_DEFAULT).vararg().arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, int64_t, ARG1_T)>(NAME)));

#define PT_REGISTER_REDUCTION_KEEPDIM_OP(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, DIM_T, DIM_DEFAULT) \
m.def(PT_REDUCTION_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).arg<DIM_T>("dim", #DIM_DEFAULT).arg<bool>("keepdim", "False").schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, DIM_T, bool)>(NAME)));

#define PT_REGISTER_REDUCTION_KEEPDIM_OP_WITH(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, DIM_T, DIM_DEFAULT, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
m.def(PT_REDUCTION_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).arg<DIM_T>("dim", #DIM_DEFAULT).arg<bool>("keepdim", "False").vararg().arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, DIM_T, bool, ARG1_T)>(NAME)));

#define PT_REGISTER_REDUCTION_KEEPDIM_OP_WITH2(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, DIM_T, DIM_DEFAULT, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
m.def(PT_REDUCTION_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).arg<DIM_T>("dim", #DIM_DEFAULT).arg<bool>("keepdim", "False").vararg().arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).arg<ARG2_T>(#ARG2_NAME, #ARG2_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, DIM_T, bool, ARG1_T, ARG2_T)>(NAME)));

#define PT_REGISTER_REDUCTION_DIM_OP_(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, DIM_T, DIM_DEFAULT) \
m.def(PT_REDUCTION_SCHEMA_BUILDER(NAME##_, OVERLOAD_NAME, POSTFIX, SELF_T).arg<DIM_T>("dim", #DIM_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, DIM_T)>(NAME##_)));

#define PT_REGISTER_REDUCTION_DIM_OP__WITH(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, DIM_T, DIM_DEFAULT, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
m.def(PT_REDUCTION_SCHEMA_BUILDER(NAME##_, OVERLOAD_NAME, POSTFIX, SELF_T).arg<int64_t>("dim", #DIM_DEFAULT).vararg().arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, DIM_T, ARG1_T)>(NAME##_)));

#define PT_REGISTER_REDUCTION_INDICES_DIM_OP(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, DIM_T, DIM_DEFAULT) \
m.def(PT_MINMAX_DIM_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).arg<DIM_T>("dim", #DIM_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<masked_pair_with_indices (*)(SELF_T, DIM_T)>(NAME)));

#define PT_REGISTER_REDUCTION_INDICES_KEEPDIM_OP(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, DIM_T, DIM_DEFAULT) \
m.def(PT_MINMAX_DIM_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).arg<DIM_T>("dim", #DIM_DEFAULT).arg<bool>("keepdim", "False").schema().c_str(), \
    TORCH_FN(static_cast<masked_pair_with_indices (*)(SELF_T, DIM_T, bool)>(NAME)));

#define PT_REGISTER_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX) \
PT_REGISTER_REDUCTION_OP(NAME, MaskedPair, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_REGISTER_REDUCTION_OP(NAME, Tensor, POSTFIX, const at::Tensor &)

#define PT_REGISTER_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_REDUCTION_OP_WITH(NAME, MaskedPair, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_REDUCTION_OP_WITH(NAME, Tensor, POSTFIX, const at::Tensor &, ARG1_T, ARG1_NAME, ARG1_DEFAULT)

#define PT_REGISTER_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_REDUCTION_OP_WITH2(NAME, MaskedPair, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_REDUCTION_OP_WITH2(NAME, Tensor, POSTFIX, const at::Tensor &, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

#define PT_REGISTER_REDUCTION_DIM_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, DIM_T, DIM_DEFAULT) \
PT_REGISTER_REDUCTION_DIM_OP(NAME, MaskedPair_dim, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, DIM_T, DIM_DEFAULT) \
PT_REGISTER_REDUCTION_DIM_OP(NAME, Tensor_dim, POSTFIX, const at::Tensor &, DIM_T, DIM_DEFAULT)

#define PT_REGISTER_REDUCTION_DIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, DIM_T, DIM_DEFAULT, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_REDUCTION_DIM_OP_WITH(NAME, MaskedPair_dim, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, DIM_T, DIM_DEFAULT, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_REDUCTION_DIM_OP_WITH(NAME, Tensor_dim, POSTFIX, const at::Tensor &, DIM_T, DIM_DEFAULT, ARG1_T, ARG1_NAME, ARG1_DEFAULT)

#define PT_REGISTER_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, DIM_T, DIM_DEFAULT) \
PT_REGISTER_REDUCTION_KEEPDIM_OP(NAME, MaskedPair_dim_IntList, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, DIM_T, DIM_DEFAULT) \
PT_REGISTER_REDUCTION_KEEPDIM_OP(NAME, Tensor_dim_IntList, POSTFIX, const at::Tensor &, DIM_T, DIM_DEFAULT)

#define PT_REGISTER_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, DIM_T, DIM_DEFAULT, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_REDUCTION_KEEPDIM_OP_WITH(NAME, MaskedPair_dim_IntList, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, DIM_T, DIM_DEFAULT, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_REDUCTION_KEEPDIM_OP_WITH(NAME, Tensor_dim_IntList, POSTFIX, const at::Tensor &, DIM_T, DIM_DEFAULT, ARG1_T, ARG1_NAME, ARG1_DEFAULT)

#define PT_REGISTER_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, DIM_T, DIM_DEFAULT, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_REDUCTION_KEEPDIM_OP_WITH2(NAME, MaskedPair_dim_IntList, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, DIM_T, DIM_DEFAULT, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_REDUCTION_KEEPDIM_OP_WITH2(NAME, Tensor_dim_IntList, POSTFIX, const at::Tensor &, DIM_T, DIM_DEFAULT, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

#define PT_REGISTER_REDUCTION_DIM_OPS__FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, DIM_T, DIM_DEFAULT) \
PT_REGISTER_REDUCTION_DIM_OP_(NAME, MaskedPair_dim, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, DIM_T, DIM_DEFAULT) \
PT_REGISTER_REDUCTION_DIM_OP_(NAME, Tensor_dim, POSTFIX, at::Tensor &, DIM_T, DIM_DEFAULT)

#define PT_REGISTER_REDUCTION_DIM_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, DIM_T, DIM_DEFAULT, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_REDUCTION_DIM_OP__WITH(NAME, MaskedPair_dim, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, DIM_T, DIM_DEFAULT, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_REDUCTION_DIM_OP__WITH(NAME, Tensor_dim, POSTFIX, at::Tensor &, DIM_T, DIM_DEFAULT, ARG1_T, ARG1_NAME, ARG1_DEFAULT)

#define PT_REGISTER_REDUCTION_AND_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, DIM_T, DIM_DEFAULT) \
PT_REGISTER_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX)                                         \
PT_REGISTER_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, DIM_T, DIM_DEFAULT)

#define PT_REGISTER_REDUCTION_AND_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, DIM_T, DIM_DEFAULT, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)                                         \
PT_REGISTER_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, DIM_T, DIM_DEFAULT, ARG1_T, ARG1_NAME, ARG1_DEFAULT)

#define PT_REGISTER_REDUCTION_AND_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, DIM_T, DIM_DEFAULT, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                                         \
PT_REGISTER_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, DIM_T, DIM_DEFAULT, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

#define PT_REGISTER_REDUCTION_DIM_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, DIM_T, DIM_DEFAULT) \
PT_REGISTER_REDUCTION_DIM_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, DIM_T, DIM_DEFAULT)                  \
PT_REGISTER_REDUCTION_DIM_OPS__FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, DIM_T, DIM_DEFAULT)

#define PT_REGISTER_REDUCTION_DIM_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, DIM_T, DIM_DEFAULT, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_REDUCTION_DIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, DIM_T, DIM_DEFAULT, ARG1_T, ARG1_NAME, ARG1_DEFAULT)                  \
PT_REGISTER_REDUCTION_DIM_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, DIM_T, DIM_DEFAULT, ARG1_T, ARG1_NAME, ARG1_DEFAULT)

#define PT_REGISTER_REDUCTION_INDICES_DIM_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, DIM_T, DIM_DEFAULT) \
PT_REGISTER_REDUCTION_INDICES_DIM_OP(NAME, MaskedPair_dim, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, DIM_T, DIM_DEFAULT) \
PT_REGISTER_REDUCTION_INDICES_DIM_OP(NAME, Tensor_dim, POSTFIX, const at::Tensor &, DIM_T, DIM_DEFAULT)

#define PT_REGISTER_REDUCTION_INDICES_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, DIM_T, DIM_DEFAULT) \
PT_REGISTER_REDUCTION_INDICES_KEEPDIM_OP(NAME, MaskedPair_dim, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, DIM_T, DIM_DEFAULT) \
PT_REGISTER_REDUCTION_INDICES_KEEPDIM_OP(NAME, Tensor_dim, POSTFIX, const at::Tensor &, DIM_T, DIM_DEFAULT)

namespace partialtorch {
    namespace ops {
        namespace impl {
            template<bool reduce = true, typename op_T, typename fill_identity_op_T,
                    typename self_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> reduction_impl(
                    op_T &&op,
                    fill_identity_op_T &&fill_identity_op,
                    const self_T &self,
                    Args &&... args) {
                at::Tensor output_data;
                if constexpr (!std::is_same_v<std::base_t<fill_identity_op_T>, nullptr_t>) {
                    output_data = op.call(fill_identity_op.call(self), args...);
                } else {
                    output_data = op.call(utils::get_data(self), args...);
                }
                c10::optional<at::Tensor> output_mask;
                if constexpr (reduce) {
                    output_mask = utils::any(utils::get_mask(self));
                } else {
                    output_mask = utils::get_mask<true>(self);
                }
                return masked_pair(output_data, output_mask);
            }

            template<typename op__T, typename fill_identity_op__T,
                    typename self_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> reduction__impl(
                    op__T &&op_,
                    fill_identity_op__T &&fill_identity_op_,
                    self_T &self,
                    Args &&... args) {
                at::Tensor self_data;
                if constexpr (!std::is_same_v<std::base_t<fill_identity_op__T>, nullptr_t>) {
                    self_data = fill_identity_op_.call(self);
                } else {
                    self_data = utils::get_data(self);
                }
                op_.call(self_data, args...);
                return masked_pair(self);
            }

            template<bool reduce = true, typename op_T, typename fill_identity_op_T,
                    typename self_T, typename dim_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> reduction_dim_impl(
                    op_T &&op,
                    fill_identity_op_T &&fill_identity_op,
                    const self_T &self,
                    dim_T dim,
                    Args... args) {
                at::Tensor output_data;
                if constexpr (!std::is_same_v<std::base_t<fill_identity_op_T>, nullptr_t>) {
                    output_data = op.call(fill_identity_op.call(self), dim, args...);
                } else {
                    output_data = op.call(utils::get_data(self), dim, args...);
                }
                c10::optional<at::Tensor> output_mask;
                if constexpr (reduce) {
                    output_mask = utils::any(utils::get_mask(self), dim);
                } else {
                    output_mask = utils::get_mask<true>(self);
                }
                return masked_pair(output_data, output_mask);
            }

            template<bool reduce = true, typename op_T, typename fill_identity_op_T,
                    typename self_T, typename dim_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> reduction_dim_impl(
                    op_T &&op,
                    fill_identity_op_T &&fill_identity_op,
                    const self_T &self,
                    dim_T dim,
                    bool keepdim,
                    Args... args) {
                at::Tensor output_data;
                if constexpr (!std::is_same_v<std::base_t<fill_identity_op_T>, nullptr_t>) {
                    output_data = op.call(fill_identity_op.call(self), dim, keepdim, args...);
                } else {
                    output_data = op.call(utils::get_data(self), dim, keepdim, args...);
                }
                c10::optional<at::Tensor> output_mask;
                if constexpr (reduce) {
                    output_mask = utils::any(utils::get_mask(self), dim, keepdim);
                } else {
                    output_mask = utils::get_mask<true>(self);
                }
                return masked_pair(output_data, output_mask);
            }

            template<typename op__T, typename fill_identity_op__T,
                    typename self_T, typename dim_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> reduction__dim_impl(
                    op__T &&op_,
                    fill_identity_op__T &&fill_identity_op_,
                    self_T &self,
                    dim_T dim,
                    Args... args) {
                at::Tensor self_data;
                if constexpr (!std::is_same_v<std::base_t<fill_identity_op__T>, nullptr_t>) {
                    self_data = fill_identity_op_.call(self);
                } else {
                    self_data = utils::get_data(self);
                }
                op_.call(self_data, dim, args...);
                return masked_pair(self);
            }

            template<bool reduce = true, typename op_T, typename fill_identity_op_T,
                    typename self_T, typename dim_T, typename... Args>
            static C10_ALWAYS_INLINE std::tuple<c10::intrusive_ptr<TensorMaskedPair>, at::Tensor>
            reduction_indices_dim_impl(
                    op_T &&op,
                    fill_identity_op_T &&fill_identity_op,
                    const self_T &self,
                    dim_T dim,
                    Args... args) {
                std::tuple<at::Tensor, at::Tensor> output;
                if constexpr (!std::is_same_v<std::base_t<fill_identity_op_T>, nullptr_t>) {
                    output = op.call(fill_identity_op.call(self), dim, args...);
                } else {
                    output = op.call(utils::get_data(self), dim, args...);
                }
                auto output_data = std::get<0>(output);
                auto output_indices = std::get<1>(output);
                c10::optional<at::Tensor> output_mask;
                if constexpr (reduce) {
                    output_mask = utils::any(utils::get_mask(self), dim);
                } else {
                    output_mask = utils::get_mask<true>(self);
                }
                if (output_mask.has_value())
                    output_indices.masked_fill_(output_mask->logical_not(), -1);
                return std::make_tuple(masked_pair(output_data, output_mask), output_indices);
            }

            template<bool reduce = true, typename op_T, typename fill_identity_op_T,
                    typename self_T, typename dim_T, typename... Args>
            static C10_ALWAYS_INLINE std::tuple<c10::intrusive_ptr<TensorMaskedPair>, at::Tensor>
            reduction_indices_dim_impl(
                    op_T &&op,
                    fill_identity_op_T &&fill_identity_op,
                    const self_T &self,
                    dim_T dim,
                    bool keepdim,
                    Args... args) {
                std::tuple<at::Tensor, at::Tensor> output;
                if constexpr (!std::is_same_v<std::base_t<fill_identity_op_T>, nullptr_t>) {
                    output = op.call(fill_identity_op.call(self), dim, keepdim, args...);
                } else {
                    output = op.call(utils::get_data(self), dim, keepdim, args...);
                }
                auto output_data = std::get<0>(output);
                auto output_indices = std::get<1>(output);
                c10::optional<at::Tensor> output_mask;
                if constexpr (reduce) {
                    output_mask = utils::any(utils::get_mask(self), dim, keepdim);
                } else {
                    output_mask = utils::get_mask<true>(self);
                }
                if (output_mask.has_value())
                    output_indices.masked_fill_(output_mask->logical_not(), -1);
                return std::make_tuple(masked_pair(output_data, output_mask), output_indices);
            }
        }  // namespace impl

        // logical
        PT_DEFINE_REDUCTION_AND_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(
                all, at::_ops::all(), utils::all_dim_IntList(), utils::_ops::fill_identity_ones(), true,
                at::OptionalIntArrayRef)

        PT_DEFINE_REDUCTION_AND_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(
                any, at::_ops::any(), utils::any_dim_IntList(), utils::_ops::fill_identity_zeros(), true,
                at::OptionalIntArrayRef)

        // arithmetics
        PT_DEFINE_REDUCTION_AND_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                sum, at::_ops::sum(), at::_ops::sum_dim_IntList(), utils::_ops::fill_identity_zeros(), true,
                at::OptionalIntArrayRef, c10::optional<at::ScalarType>, dtype)

        PT_DEFINE_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                nansum, at::_ops::nansum(), utils::_ops::fill_identity_nan(), true,
                at::OptionalIntArrayRef, c10::optional<at::ScalarType>, dtype)

        PT_DEFINE_REDUCTION_AND_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                prod, at::_ops::prod(), utils::prod_dim_IntList(), utils::_ops::fill_identity_ones(), true,
                at::OptionalIntArrayRef, c10::optional<at::ScalarType>, dtype)

        PT_DEFINE_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(
                logsumexp, at::_ops::logsumexp(), utils::_ops::fill_identity_neginf(), true,
                at::IntArrayRef)

        PT_DEFINE_REDUCTION_DIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                softmax, at::_ops::softmax_int(), utils::_ops::fill_identity_neginf(), false,
                int64_t, c10::optional<at::ScalarType>, dtype)

        PT_DEFINE_REDUCTION_DIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                log_softmax, at::_ops::log_softmax_int(), utils::_ops::fill_identity_neginf(), false,
                int64_t, c10::optional<at::ScalarType>, dtype)

        PT_DEFINE_REDUCTION_DIM_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                cumsum, at::_ops::cumsum(), utils::_ops::fill_identity_zeros(),
                at::_ops::cumsum_(), utils::_ops::fill_identity_zeros_(),
                int64_t, c10::optional<at::ScalarType>, dtype)

        PT_DEFINE_REDUCTION_DIM_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                cumprod, at::_ops::cumprod(), utils::_ops::fill_identity_ones(),
                at::_ops::cumprod_(), utils::_ops::fill_identity_ones_(),
                int64_t, c10::optional<at::ScalarType>, dtype)

        namespace impl {
            namespace _ops {
                template<bool copy = false>
                struct fill_diagonal_identity_zeros {
                    static at::Tensor call(const c10::intrusive_ptr<TensorMaskedPair> &self) {
                        if (!utils::has_tensor_mask(self))
                            return utils::get_data<copy>(self);
                        auto output_data = utils::get_data<copy>(self);
                        auto output_mask = utils::get_tensor_mask(self);
                        auto diagonal_data = at::diagonal(output_data);
                        auto diagonal_mask = at::diagonal(output_mask);
                        diagonal_data.masked_fill_(diagonal_mask.logical_not(), 0);
                        return output_data;
                    }

                    template<typename T>
                    static T call(const T &self) {
                        return utils::get_data<copy>(self);
                    }
                };
            }
        }

        PT_DEFINE_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS(
                trace, at::_ops::trace(), impl::_ops::fill_diagonal_identity_zeros(), true)

        // statistics
        namespace impl {
            namespace _ops {
                struct nanmean_zerodim {
                    static at::Tensor call(const at::Tensor &self,
                                           c10::optional<at::ScalarType> dtype) {
                        return at::_ops::nanmean::call(self, {}, false, dtype);
                    }
                };
            }
        }

        // TODO: make fill_identity_op of mean and nanmean constexpr if possible
        c10::intrusive_ptr<TensorMaskedPair> mean(const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                                                  c10::optional<at::ScalarType> dtype) {
            static constexpr auto op = impl::_ops::nanmean_zerodim();
            auto fill_identity_op = utils::_ops::sequential(
                    utils::_ops::cast_state(dtype),
                    utils::_ops::fill_identity_nan());
            return impl::reduction_impl<true>(op, fill_identity_op, self, dtype);
        }

        c10::intrusive_ptr<TensorMaskedPair> mean(const at::Tensor &self, c10::optional<at::ScalarType> dtype) {
            static constexpr auto op = impl::_ops::nanmean_zerodim();
            auto fill_identity_op = utils::_ops::sequential(
                    utils::_ops::cast_state(dtype),
                    utils::_ops::fill_identity_nan());
            return impl::reduction_impl<true>(op, fill_identity_op, self, dtype);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        mean(const_intrusive_ptr_arg_t<TensorMaskedPair> self, at::OptionalIntArrayRef dim, bool keepdim,
             c10::optional<at::ScalarType> dtype) {
            static constexpr auto op = at::_ops::nanmean();
            auto fill_identity_op = utils::_ops::sequential(
                    utils::_ops::cast_state(dtype),
                    utils::_ops::fill_identity_nan());
            return impl::reduction_dim_impl<true>(op, fill_identity_op, self, dim, keepdim, dtype);
        }

        c10::intrusive_ptr<TensorMaskedPair> mean(const at::Tensor &self, at::OptionalIntArrayRef dim, bool keepdim,
                                                  c10::optional<at::ScalarType> dtype) {
            static constexpr auto op = at::_ops::nanmean();
            auto fill_identity_op = utils::_ops::sequential(
                    utils::_ops::cast_state(dtype),
                    utils::_ops::fill_identity_nan());
            return impl::reduction_dim_impl<true>(op, fill_identity_op, self, dim, keepdim, dtype);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        nanmean(const_intrusive_ptr_arg_t<TensorMaskedPair> self, at::OptionalIntArrayRef dim, bool keepdim,
                c10::optional<at::ScalarType> dtype) {
            static constexpr auto op = at::_ops::nanmean();
            auto fill_identity_op = utils::_ops::sequential(
                    utils::_ops::cast_state(dtype),
                    utils::_ops::fill_identity_nan());
            return impl::reduction_dim_impl<true>(op, fill_identity_op, self, dim, keepdim, dtype);
        }

        c10::intrusive_ptr<TensorMaskedPair> nanmean(const at::Tensor &self, at::OptionalIntArrayRef dim, bool keepdim,
                                                     c10::optional<at::ScalarType> dtype) {
            static constexpr auto op = at::_ops::nanmean();
            auto fill_identity_op = utils::_ops::sequential(
                    utils::_ops::cast_state(dtype),
                    utils::_ops::fill_identity_nan());
            return impl::reduction_dim_impl<true>(op, fill_identity_op, self, dim, keepdim, dtype);
        }

        c10::intrusive_ptr<TensorMaskedPair> median(const_intrusive_ptr_arg_t<TensorMaskedPair> self) {
            auto dtype = self->data_.scalar_type();
            bool needs_cast = !at::isFloatingType(dtype);
            static constexpr auto op = at::_ops::nanmedian();
            if (needs_cast) {
                auto fill_identity_op = utils::_ops::sequential(
                        utils::_ops::cast_state(at::kFloat),
                        utils::_ops::fill_identity_nan());
                auto result = impl::reduction_impl<true>(op, fill_identity_op, self);
                return utils::_ops::cast_dtype<false>::call(result, dtype);
            } else {
                static constexpr auto fill_identity_op = utils::_ops::fill_identity_nan();
                auto result = impl::reduction_impl<true>(op, fill_identity_op, self);
                return result;
            }
        }

        c10::intrusive_ptr<TensorMaskedPair> median(const at::Tensor &self) {
            auto dtype = self.scalar_type();
            bool needs_cast = !at::isFloatingType(dtype);
            static constexpr auto op = at::_ops::nanmedian();
            if (needs_cast) {
                auto fill_identity_op = utils::_ops::sequential(
                        utils::_ops::cast_state(at::kFloat),
                        utils::_ops::fill_identity_nan());
                auto result = impl::reduction_impl<true>(op, fill_identity_op, self);
                return utils::_ops::cast_dtype<false>::call(result, dtype);
            } else {
                static constexpr auto fill_identity_op = utils::_ops::fill_identity_nan();
                auto result = impl::reduction_impl<true>(op, fill_identity_op, self);
                return result;
            }
        }

        std::tuple<c10::intrusive_ptr<TensorMaskedPair>, at::Tensor>
        median(const_intrusive_ptr_arg_t<TensorMaskedPair> self, int64_t dim, bool keepdim) {
            auto dtype = self->data_.scalar_type();
            bool needs_cast = !at::isFloatingType(dtype);
            static constexpr auto op = at::_ops::nanmedian_dim();
            if (needs_cast) {
                auto fill_identity_op = utils::_ops::sequential(
                        utils::_ops::cast_state(at::kFloat),
                        utils::_ops::fill_identity_nan());
                auto result = impl::reduction_indices_dim_impl<true>(op, fill_identity_op, self, dim, keepdim);
                return std::make_tuple(utils::_ops::cast_dtype<false>::call(std::get<0>(result), dtype),
                                       std::get<1>(result));
            } else {
                static constexpr auto fill_identity_op = utils::_ops::fill_identity_nan();
                auto result = impl::reduction_indices_dim_impl<true>(op, fill_identity_op, self, dim, keepdim);
                return result;
            }
        }

        std::tuple<c10::intrusive_ptr<TensorMaskedPair>, at::Tensor>
        median(const at::Tensor &self, int64_t dim, bool keepdim) {
            auto dtype = self.scalar_type();
            bool needs_cast = !at::isFloatingType(dtype);
            static constexpr auto op = at::_ops::nanmedian_dim();
            if (needs_cast) {
                auto fill_identity_op = utils::_ops::sequential(
                        utils::_ops::cast_state(at::kFloat),
                        utils::_ops::fill_identity_nan());
                auto result = impl::reduction_indices_dim_impl<true>(op, fill_identity_op, self, dim, keepdim);
                return std::make_tuple(utils::_ops::cast_dtype<false>::call(std::get<0>(result), dtype),
                                       std::get<1>(result));
            } else {
                static constexpr auto fill_identity_op = utils::_ops::fill_identity_nan();
                auto result = impl::reduction_indices_dim_impl<true>(op, fill_identity_op, self, dim, keepdim);
                return result;
            }
        }

        PT_DEFINE_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS(
                nanmedian, at::_ops::nanmedian(), utils::_ops::fill_identity_nan(), true)

        PT_DEFINE_REDUCTION_INDICES_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(
                nanmedian, at::_ops::nanmedian_dim(), utils::_ops::fill_identity_nan(), true,
                int64_t)

        namespace impl {
            template<typename self_T, typename dim_T>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> norm_impl(
                    const self_T &self,
                    const c10::optional<at::Scalar> &p,
                    dim_T &&dim,
                    bool keepdim,
                    c10::optional<at::ScalarType> dtype) {
                at::Tensor output_data;
                if (p.has_value() && p.value().to<double>() < 0) {
                    output_data = at::_ops::norm_ScalarOpt_dim_dtype::call(
                            utils::_ops::fill_identity_posinf<false>::call(
                                    utils::_ops::cast_dtype<false>::call(self, dtype)),
                            p, dim, keepdim, dtype.value_or(utils::get_data(self).scalar_type()));
                } else {
                    output_data = at::_ops::norm_ScalarOpt_dim_dtype::call(
                            utils::_ops::fill_identity_zeros<false>::call(self), p, dim, keepdim,
                            dtype.value_or(utils::get_data(self).scalar_type()));
                }
                auto output_mask = utils::any(utils::get_mask(self), dim, keepdim);
                return masked_pair(output_data, output_mask);
            }

            template<typename op_T, typename self_T, typename ord_T, typename dim_T>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> linalg_norm_impl(
                    const self_T &self,
                    ord_T &&ord,
                    dim_T &&dim,
                    bool keepdim,
                    c10::optional<at::ScalarType> dtype) {
                at::Tensor output_data;
                if constexpr (std::is_same_v<c10::base_t<ord_T>, at::Scalar>) {
                    if (ord.template to<double>() < 0) {
                        output_data = op_T::call(
                                utils::_ops::fill_identity_posinf<false>::call(
                                        utils::_ops::cast_dtype<false>::call(self, dtype)),
                                ord, dim, keepdim, dtype.value_or(utils::get_data(self).scalar_type()));
                    } else {
                        output_data = op_T::call(
                                utils::_ops::fill_identity_zeros<false>::call(self), ord, dim, keepdim,
                                dtype.value_or(utils::get_data(self).scalar_type()));
                    }
                } else {
                    output_data = op_T::call(
                            utils::_ops::fill_identity_zeros<false>::call(self), ord, dim, keepdim,
                            dtype.value_or(utils::get_data(self).scalar_type()));
                }
                auto output_mask = utils::any(utils::get_mask(self), dim, keepdim);
                return masked_pair(output_data, output_mask);
            }
        }

        c10::intrusive_ptr<TensorMaskedPair> norm(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                const c10::optional<at::Scalar> &p,
                at::IntArrayRef dim,
                bool keepdim,
                c10::optional<at::ScalarType> dtype) {
            return impl::norm_impl(self, p, dim, keepdim, dtype);
        }

        c10::intrusive_ptr<TensorMaskedPair> norm(
                const at::Tensor &self,
                const c10::optional<at::Scalar> &p,
                at::IntArrayRef dim,
                bool keepdim,
                c10::optional<at::ScalarType> dtype) {
            return impl::norm_impl(self, p, dim, keepdim, dtype);
        }

        c10::intrusive_ptr<TensorMaskedPair> linalg_norm(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                const at::Scalar &ord,
                at::OptionalIntArrayRef dim,
                bool keepdim,
                c10::optional<at::ScalarType> dtype) {
            return impl::linalg_norm_impl<at::_ops::linalg_norm>(self, ord, dim, keepdim, dtype);
        }

        c10::intrusive_ptr<TensorMaskedPair> linalg_norm(
                const at::Tensor &self,
                const at::Scalar &ord,
                at::OptionalIntArrayRef dim,
                bool keepdim,
                c10::optional<at::ScalarType> dtype) {
            return impl::linalg_norm_impl<at::_ops::linalg_norm>(self, ord, dim, keepdim, dtype);
        }

        c10::intrusive_ptr<TensorMaskedPair> linalg_norm(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                c10::string_view ord,
                at::OptionalIntArrayRef dim,
                bool keepdim,
                c10::optional<at::ScalarType> dtype) {
            return impl::linalg_norm_impl<at::_ops::linalg_norm_ord_str>(self, ord, dim, keepdim, dtype);
        }

        c10::intrusive_ptr<TensorMaskedPair> linalg_norm(
                const at::Tensor &self,
                c10::string_view ord,
                at::OptionalIntArrayRef dim,
                bool keepdim,
                c10::optional<at::ScalarType> dtype) {
            return impl::linalg_norm_impl<at::_ops::linalg_norm_ord_str>(self, ord, dim, keepdim, dtype);
        }

        c10::intrusive_ptr<TensorMaskedPair> linalg_vector_norm(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                const at::Scalar &ord,
                at::OptionalIntArrayRef dim,
                bool keepdim,
                c10::optional<at::ScalarType> dtype) {
            return impl::linalg_norm_impl<at::_ops::linalg_vector_norm>(self, ord, dim, keepdim, dtype);
        }

        c10::intrusive_ptr<TensorMaskedPair> linalg_vector_norm(
                const at::Tensor &self,
                const at::Scalar &ord,
                at::OptionalIntArrayRef dim,
                bool keepdim,
                c10::optional<at::ScalarType> dtype) {
            return impl::linalg_norm_impl<at::_ops::linalg_vector_norm>(self, ord, dim, keepdim, dtype);
        }

        c10::intrusive_ptr<TensorMaskedPair> linalg_matrix_norm(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                const at::Scalar &ord,
                at::IntArrayRef dim,
                bool keepdim,
                c10::optional<at::ScalarType> dtype) {
            return impl::linalg_norm_impl<at::_ops::linalg_matrix_norm>(self, ord, dim, keepdim, dtype);
        }

        c10::intrusive_ptr<TensorMaskedPair> linalg_matrix_norm(
                const at::Tensor &self,
                const at::Scalar &ord,
                at::IntArrayRef dim,
                bool keepdim,
                c10::optional<at::ScalarType> dtype) {
            return impl::linalg_norm_impl<at::_ops::linalg_matrix_norm>(self, ord, dim, keepdim, dtype);
        }

        c10::intrusive_ptr<TensorMaskedPair> linalg_matrix_norm(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                c10::string_view ord,
                at::IntArrayRef dim,
                bool keepdim,
                c10::optional<at::ScalarType> dtype) {
            return impl::linalg_norm_impl<at::_ops::linalg_matrix_norm_str_ord>(self, ord, dim, keepdim, dtype);
        }

        c10::intrusive_ptr<TensorMaskedPair> linalg_matrix_norm(
                const at::Tensor &self,
                c10::string_view ord,
                at::IntArrayRef dim,
                bool keepdim,
                c10::optional<at::ScalarType> dtype) {
            return impl::linalg_norm_impl<at::_ops::linalg_matrix_norm_str_ord>(self, ord, dim, keepdim, dtype);
        }

        namespace impl {
            template<bool sqrt, typename self_T, typename correction_T>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> stdvar_impl(
                    const self_T &self,
                    const at::OptionalIntArrayRef &dim,
                    correction_T &&unbiased_or_correction,
                    bool keepdim) {
                if (!utils::get_mask(self).has_value()) {
                    if constexpr (std::is_same_v<c10::base_t<correction_T>, at::Scalar>) {
                        if constexpr (sqrt) {
                            return masked_pair(at::_ops::std_correction::call(
                                    utils::get_data(self), dim, unbiased_or_correction, keepdim));
                        } else {
                            return masked_pair(at::_ops::var_correction::call(
                                    utils::get_data(self), dim, unbiased_or_correction, keepdim));
                        }
                    } else {  // unbiased
                        if constexpr (sqrt) {
                            return masked_pair(at::_ops::std_dim::call(
                                    utils::get_data(self), dim, unbiased_or_correction, keepdim));
                        } else {
                            return masked_pair(at::_ops::var_dim::call(
                                    utils::get_data(self), dim, unbiased_or_correction, keepdim));
                        }
                    }
                }
                at::Scalar correction_val;
                if constexpr (std::is_same_v<c10::base_t<correction_T>, at::Scalar>) {
                    correction_val = unbiased_or_correction.value_or(1);
                } else {  // unbiased
                    correction_val = unbiased_or_correction;
                }

                auto mask = utils::get_mask(self).value();
                auto not_mask = mask.logical_not();
                auto sample_total = at::sum(utils::get_data(self).masked_fill(not_mask, 0), dim, true);
                auto count = at::sum(mask, dim, true, sample_total.scalar_type());
                auto x = utils::get_data(self) - sample_total.div_(count);
                x = x.masked_fill_(not_mask, 0);
                auto total = at::sum(x * x.conj(), dim, keepdim);
                if (!keepdim)
                    count = count.view_as(total);
                if (!correction_val.equal(0))
                    count.sub_(correction_val).clamp_min_(0);
                auto output_data = total.div_(count);
                if constexpr (sqrt) {
                    output_data = output_data.sqrt_();
                }
                auto output_mask = utils::any(mask, dim, keepdim);
                return masked_pair(output_data, output_mask);
            }
        }

        c10::intrusive_ptr<TensorMaskedPair> var(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                bool unbiased) {
            return impl::stdvar_impl<false>(self, {}, unbiased, false);
        }

        c10::intrusive_ptr<TensorMaskedPair> var(
                const at::Tensor &self,
                bool unbiased) {
            return impl::stdvar_impl<false>(self, {}, unbiased, false);
        }

        c10::intrusive_ptr<TensorMaskedPair> var(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                at::OptionalIntArrayRef dim,
                bool unbiased,
                bool keepdim) {
            return impl::stdvar_impl<false>(self, dim, unbiased, keepdim);
        }

        c10::intrusive_ptr<TensorMaskedPair> var(
                const at::Tensor &self,
                at::OptionalIntArrayRef dim,
                bool unbiased,
                bool keepdim) {
            return impl::stdvar_impl<false>(self, dim, unbiased, keepdim);
        }

        c10::intrusive_ptr<TensorMaskedPair> var(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                at::OptionalIntArrayRef dim,
                const c10::optional<at::Scalar> &correction,
                bool keepdim) {
            return impl::stdvar_impl<false>(self, dim, correction, keepdim);
        }

        c10::intrusive_ptr<TensorMaskedPair> var(
                const at::Tensor &self,
                at::OptionalIntArrayRef dim,
                const c10::optional<at::Scalar> &correction,
                bool keepdim) {
            return impl::stdvar_impl<false>(self, dim, correction, keepdim);
        }

        c10::intrusive_ptr<TensorMaskedPair> std(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                bool unbiased) {
            return impl::stdvar_impl<true>(self, {}, unbiased, false);
        }

        c10::intrusive_ptr<TensorMaskedPair> std(
                const at::Tensor &self,
                bool unbiased) {
            return impl::stdvar_impl<true>(self, {}, unbiased, false);
        }

        c10::intrusive_ptr<TensorMaskedPair> std(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                at::OptionalIntArrayRef dim,
                bool unbiased,
                bool keepdim) {
            return impl::stdvar_impl<true>(self, dim, unbiased, keepdim);
        }

        c10::intrusive_ptr<TensorMaskedPair> std(
                const at::Tensor &self,
                at::OptionalIntArrayRef dim,
                bool unbiased,
                bool keepdim) {
            return impl::stdvar_impl<true>(self, dim, unbiased, keepdim);
        }

        c10::intrusive_ptr<TensorMaskedPair> std(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                at::OptionalIntArrayRef dim,
                const c10::optional<at::Scalar> &correction,
                bool keepdim) {
            return impl::stdvar_impl<true>(self, dim, correction, keepdim);
        }

        c10::intrusive_ptr<TensorMaskedPair> std(
                const at::Tensor &self,
                at::OptionalIntArrayRef dim,
                const c10::optional<at::Scalar> &correction,
                bool keepdim) {
            return impl::stdvar_impl<true>(self, dim, correction, keepdim);
        }

        // min max
        PT_DEFINE_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS(
                min, at::_ops::min(), utils::_ops::fill_identity_posinf(), true)

        PT_DEFINE_REDUCTION_INDICES_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(
                min, at::_ops::min_dim(), utils::_ops::fill_identity_posinf(), true, int64_t)

        PT_DEFINE_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS(
                max, at::_ops::max(), utils::_ops::fill_identity_neginf(), true)

        PT_DEFINE_REDUCTION_INDICES_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(
                max, at::_ops::max_dim(), utils::_ops::fill_identity_neginf(), true, int64_t)

        PT_DEFINE_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(
                amin, at::_ops::amin(), utils::_ops::fill_identity_posinf(), true, at::IntArrayRef)

        PT_DEFINE_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(
                amax, at::_ops::amax(), utils::_ops::fill_identity_neginf(), true, at::IntArrayRef)

        namespace impl {
            template<typename op_T, typename fill_identity_op_T,
                    typename self_T, typename dim_T, typename... Args>
            static C10_ALWAYS_INLINE at::Tensor argminmax_impl(
                    op_T &&op,
                    const fill_identity_op_T &fill_identity_op,
                    const self_T &self,
                    dim_T dim,
                    bool keepdim,
                    Args... args) {
                at::Tensor output_indices;
                if constexpr (!std::is_same_v<std::base_t<fill_identity_op_T>, nullptr_t>) {
                    output_indices = op.call(fill_identity_op.call(self), dim, keepdim, args...);
                } else {
                    output_indices = op.call(utils::get_data(self), dim, keepdim, args...);
                }
                c10::optional<at::Tensor> output_mask;
                if constexpr (std::is_same_v<dim_T, c10::optional<int64_t>>) {
                    at::OptionalIntArrayRef dim_ = dim.has_value() ? dim.value()
                                                                   : at::OptionalIntArrayRef{};
                    output_mask = utils::any(utils::get_mask(self), dim_, keepdim);
                } else {
                    output_mask = utils::any(utils::get_mask(self), dim, keepdim);
                }
                if (output_mask.has_value())
                    output_indices.masked_fill_(output_mask->logical_not(), -1);
                return output_indices;
            }
        }

        at::Tensor argmin(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                c10::optional<int64_t> dim,
                bool keepdim) {
            static constexpr auto op = at::_ops::argmin();
            static constexpr auto fill_identity_op = utils::_ops::fill_identity_posinf();
            return impl::argminmax_impl(op, fill_identity_op, self, dim, keepdim);
        }

        at::Tensor argmin(
                const at::Tensor &self,
                c10::optional<int64_t> dim,
                bool keepdim) {
            static constexpr auto op = at::_ops::argmin();
            static constexpr auto fill_identity_op = utils::_ops::fill_identity_posinf();
            return impl::argminmax_impl(op, fill_identity_op, self, dim, keepdim);
        }

        at::Tensor argmax(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                c10::optional<int64_t> dim,
                bool keepdim) {
            static constexpr auto op = at::_ops::argmax();
            static constexpr auto fill_identity_op = utils::_ops::fill_identity_neginf();
            return impl::argminmax_impl(op, fill_identity_op, self, dim, keepdim);
        }

        at::Tensor argmax(
                const at::Tensor &self,
                c10::optional<int64_t> dim,
                bool keepdim) {
            static constexpr auto op = at::_ops::argmax();
            static constexpr auto fill_identity_op = utils::_ops::fill_identity_neginf();
            return impl::argminmax_impl(op, fill_identity_op, self, dim, keepdim);
        }

        PT_DEFINE_REDUCTION_INDICES_DIM_OPS_FORALL_TENSOR_OVERLOADS(
                cummin, at::_ops::cummin(), utils::_ops::fill_identity_posinf(), false,
                int64_t)

        PT_DEFINE_REDUCTION_INDICES_DIM_OPS_FORALL_TENSOR_OVERLOADS(
                cummax, at::_ops::cummax(), utils::_ops::fill_identity_neginf(), false,
                int64_t)

        // torch.nn.functional
        namespace impl {
            template<typename self_T>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> softmin_impl(
                    const self_T &self,
                    int64_t dim,
                    c10::optional<at::ScalarType> dtype) {
                return softmax(masked_pair(-utils::get_data(self), utils::get_mask(self)), dim, dtype);
            }

            template<typename self_T>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> normalize_impl(
                    const self_T &self,
                    const c10::optional<at::Scalar> &p,
                    int64_t dim,
                    double eps) {
                auto self_data = utils::get_data(self);
                auto denom = utils::get_data(norm(self, p, dim, true, {})).clamp_min(eps).expand_as(self_data);
                auto output_data = at::div(self_data, denom);
                auto output_mask = utils::get_mask<true>(self);
                return masked_pair(output_data, output_mask);
            }
        }

        c10::intrusive_ptr<TensorMaskedPair> softmin(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                int64_t dim,
                c10::optional<at::ScalarType> dtype) {
            return impl::softmin_impl(self, dim, dtype);
        }

        c10::intrusive_ptr<TensorMaskedPair> softmin(
                const at::Tensor &self,
                int64_t dim,
                c10::optional<at::ScalarType> dtype) {
            return impl::softmin_impl(self, dim, dtype);
        }

        c10::intrusive_ptr<TensorMaskedPair> normalize(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                const c10::optional<at::Scalar> &p,
                int64_t dim,
                double eps) {
            return impl::normalize_impl(self, p, dim, eps);
        }

        c10::intrusive_ptr<TensorMaskedPair> normalize(
                const at::Tensor &self,
                const c10::optional<at::Scalar> &p,
                int64_t dim,
                double eps) {
            return impl::normalize_impl(self, p, dim, eps);
        }

        // scaled arithmetics
        namespace impl {
            template<typename op_T, typename fill_identity_op_T, typename canonical_scale_T,
                    typename self_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> scaled_reduction_impl(
                    op_T &&op,
                    fill_identity_op_T &&fill_identity_op,
                    canonical_scale_T &&canonical_scale,
                    const self_T &self,
                    Args &&... args) {
                auto output_data = op.call(utils::get_data(self), args...);
                auto mask_ratio_options = output_data.options();
                at::Tensor mask_ratio;
                {
                    at::NoGradGuard g;
                    mask_ratio = at::sum(
                            utils::get_tensor_mask(self, mask_ratio_options));
                    if constexpr (std::is_same_v<std::base_t<canonical_scale_T>, nullptr_t>) {
                        utils::izero_ldiv_(mask_ratio, utils::get_data(self).numel());
                    } else if constexpr (std::is_same_v<std::base_t<canonical_scale_T>, at::Scalar> ||
                                         std::is_same_v<std::base_t<canonical_scale_T>, at::Tensor> ||
                                         std::is_arithmetic_v<std::base_t<canonical_scale_T>>) {
                        utils::izero_ldiv_(mask_ratio, canonical_scale);
                    } else {
                        utils::izero_ldiv_(
                                mask_ratio,
                                canonical_scale.call(utils::get_data(self), args...)
                        );
                    }
                }
                output_data.mul_(mask_ratio);
                auto output_mask = utils::any(utils::get_mask(self));
                return masked_pair(output_data, output_mask);
            }

            template<typename op_T, typename fill_identity_op_T, typename canonical_scale_T,
                    typename self_T, typename dim_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> scaled_reduction_dim_impl(
                    op_T &&op,
                    fill_identity_op_T &&fill_identity_op,
                    canonical_scale_T &&canonical_scale,
                    const self_T &self,
                    dim_T dim,
                    Args... args) {
                auto output_data = op.call(fill_identity_op.call(self), dim, args...);
                auto mask_ratio_options = output_data.options();
                at::Tensor mask_ratio;
                {
                    at::NoGradGuard g;
                    mask_ratio = at::sum(
                            utils::get_tensor_mask(self, mask_ratio_options), dim);
                    if constexpr (std::is_same_v<std::base_t<canonical_scale_T>, nullptr_t>) {
                        utils::izero_ldiv_(mask_ratio, utils::get_data(self).numel() / mask_ratio.numel());
                    } else if constexpr (std::is_same_v<std::base_t<canonical_scale_T>, at::Scalar> ||
                                         std::is_same_v<std::base_t<canonical_scale_T>, at::Tensor> ||
                                         std::is_arithmetic_v<std::base_t<canonical_scale_T>>) {
                        utils::izero_ldiv_(mask_ratio, canonical_scale);
                    } else {
                        utils::izero_ldiv_(
                                mask_ratio,
                                canonical_scale.call(utils::get_data(self), args...)
                        );
                    }
                }
                output_data.mul_(mask_ratio);
                auto output_mask = utils::any(utils::get_mask(self), dim);
                return masked_pair(output_data, output_mask);
            }

            template<typename op_T, typename fill_identity_op_T, typename canonical_scale_T,
                    typename self_T, typename dim_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> scaled_reduction_dim_impl(
                    op_T &&op,
                    fill_identity_op_T &&fill_identity_op,
                    canonical_scale_T &&canonical_scale,
                    const self_T &self,
                    dim_T dim,
                    bool keepdim,
                    Args... args) {
                auto output_data = op.call(fill_identity_op.call(self), dim, keepdim, args...);
                auto mask_ratio_options = output_data.options();
                at::Tensor mask_ratio;
                {
                    at::NoGradGuard g;
                    mask_ratio = at::sum(
                            utils::get_tensor_mask(self, mask_ratio_options), dim, keepdim);
                    if constexpr (std::is_same_v<std::base_t<canonical_scale_T>, nullptr_t>) {
                        utils::izero_ldiv_(mask_ratio, utils::get_data(self).numel() / mask_ratio.numel());
                    } else if constexpr (std::is_same_v<std::base_t<canonical_scale_T>, at::Scalar> ||
                                         std::is_same_v<std::base_t<canonical_scale_T>, at::Tensor> ||
                                         std::is_arithmetic_v<std::base_t<canonical_scale_T>>) {
                        utils::izero_ldiv_(mask_ratio, canonical_scale);
                    } else {
                        utils::izero_ldiv_(
                                mask_ratio,
                                canonical_scale.call(utils::get_data(self), args...)
                        );
                    }
                }
                output_data.mul_(mask_ratio);
                auto output_mask = utils::any(utils::get_mask(self), dim, keepdim);
                return masked_pair(output_data, output_mask);
            }

            template<typename op_T, typename fill_identity_op_T, typename canonical_scale_T,
                    typename self_T, typename dim_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> scaled_reduction_dim_impl(
                    op_T &&op,
                    fill_identity_op_T &&fill_identity_op,
                    canonical_scale_T &&canonical_scale,
                    const self_T &self,
                    dim_T dim,
                    bool keepdim,
                    c10::optional<at::ScalarType> dtype,
                    Args... args) {
                auto output_data = op.call(fill_identity_op.call(self), dim, keepdim, dtype, args...);
                auto mask_ratio_options = output_data.options();
                at::Tensor mask_ratio;
                {
                    at::NoGradGuard g;
                    mask_ratio = at::sum(
                            utils::get_tensor_mask(self, mask_ratio_options), dim, keepdim, dtype);
                    if constexpr (std::is_same_v<std::base_t<canonical_scale_T>, nullptr_t>) {
                        utils::izero_ldiv_(mask_ratio, utils::get_data(self).numel() / mask_ratio.numel());
                    } else if constexpr (std::is_same_v<std::base_t<canonical_scale_T>, at::Scalar> ||
                                         std::is_same_v<std::base_t<canonical_scale_T>, at::Tensor> ||
                                         std::is_arithmetic_v<std::base_t<canonical_scale_T>>) {
                        utils::izero_ldiv_(mask_ratio, canonical_scale);
                    } else {
                        utils::izero_ldiv_(
                                mask_ratio,
                                canonical_scale.call(utils::get_data(self), args...)
                        );
                    }
                }
                output_data.mul_(mask_ratio);
                auto output_mask = utils::any(utils::get_mask(self), dim, keepdim);
                return masked_pair(output_data, output_mask);
            }
        }  // namespace impl

        PT_DEFINE_SCALED_REDUCTION_AND_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                sum, at::_ops::sum(), at::_ops::sum_dim_IntList(), utils::_ops::fill_identity_zeros(), nullptr,
                at::OptionalIntArrayRef, c10::optional<at::ScalarType>, dtype)

        PT_DEFINE_SCALED_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                nansum, at::_ops::nansum(), utils::_ops::fill_identity_nan(), nullptr,
                at::OptionalIntArrayRef, c10::optional<at::ScalarType>, dtype)

        namespace impl {
            namespace _ops {
                struct trace_scale {
                    static inline int64_t call(const at::Tensor &self) {
                        return std::min(self.size(0), self.size(1));
                    }
                };
            }
        }

        PT_DEFINE_SCALED_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS(
                trace, at::_ops::trace(), utils::_ops::fill_identity_zeros(), impl::_ops::trace_scale())

        TORCH_LIBRARY_FRAGMENT(partialtorch, m) {
            // logical
            PT_REGISTER_REDUCTION_AND_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(
                    all, , at::OptionalIntArrayRef,)
            PT_REGISTER_REDUCTION_AND_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(
                    any, , at::OptionalIntArrayRef,)

            // arithmetics
            PT_REGISTER_REDUCTION_AND_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    sum, ,
                    at::OptionalIntArrayRef, , c10::optional<at::ScalarType>, dtype, None)
            PT_REGISTER_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    nansum, ,
                    at::OptionalIntArrayRef, , c10::optional<at::ScalarType>, dtype, None)
            PT_REGISTER_REDUCTION_AND_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    prod, ,
                    at::OptionalIntArrayRef, , c10::optional<at::ScalarType>, dtype, None)
            PT_REGISTER_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(
                    logsumexp, , at::IntArrayRef,)
            PT_REGISTER_REDUCTION_DIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    softmax, ,
                    int64_t, , c10::optional<at::ScalarType>, dtype, None)
            PT_REGISTER_REDUCTION_DIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    log_softmax, ,
                    int64_t, , c10::optional<at::ScalarType>, dtype, None)
            PT_REGISTER_REDUCTION_DIM_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                    cumsum, ,
                    int64_t, , c10::optional<at::ScalarType>, dtype, None)
            PT_REGISTER_REDUCTION_DIM_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                    cumprod, ,
                    int64_t, , c10::optional<at::ScalarType>, dtype, None)
            PT_REGISTER_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS(
                    trace,)

            // statistics
            PT_REGISTER_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    mean, , c10::optional<at::ScalarType>, dtype, None)
            PT_REGISTER_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    mean, ,
                    at::OptionalIntArrayRef, , c10::optional<at::ScalarType>, dtype, None)
            PT_REGISTER_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    nanmean, ,
                    at::OptionalIntArrayRef, , c10::optional<at::ScalarType>, dtype, None)
            PT_REGISTER_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS(median,)
            PT_REGISTER_REDUCTION_INDICES_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(
                    median, , int64_t,)
            PT_REGISTER_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS(nanmedian,)
            PT_REGISTER_REDUCTION_INDICES_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(
                    nanmedian, , int64_t,)

            m.def(utils::FunctionSchemaBuilder("var").add_overload("MaskedPair")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .arg<bool>("unbiased", "True")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          bool)>(var)));
            m.def(utils::FunctionSchemaBuilder("var").add_overload("Tensor")
                          .arg<const at::Tensor &>("self")
                          .arg<bool>("unbiased", "True")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          bool)>(var)));

            m.def(utils::FunctionSchemaBuilder("var").add_overload("MaskedPair").add_overload("dim")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .arg<at::OptionalIntArrayRef>("dim")
                          .arg<bool>("unbiased", "True")
                          .arg<bool>("keepdim", "False")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          at::OptionalIntArrayRef,
                          bool, bool)>(var)));
            m.def(utils::FunctionSchemaBuilder("var").add_overload("Tensor").add_overload("dim")
                          .arg<const at::Tensor &>("self")
                          .arg<at::OptionalIntArrayRef>("dim")
                          .arg<bool>("unbiased", "True")
                          .arg<bool>("keepdim", "False")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          at::OptionalIntArrayRef,
                          bool, bool)>(var)));

            m.def(utils::FunctionSchemaBuilder("var").add_overload("MaskedPair").add_overload("correction")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .arg<at::OptionalIntArrayRef>("dim", "None")
                          .arg<const c10::optional<at::Scalar> &>("correction", "None")
                          .arg<bool>("keepdim", "False")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          at::OptionalIntArrayRef,
                          const c10::optional<at::Scalar> &, bool)>(var)));
            m.def(utils::FunctionSchemaBuilder("var").add_overload("Tensor").add_overload("correction")
                          .arg<const at::Tensor &>("self")
                          .arg<at::OptionalIntArrayRef>("dim", "None")
                          .arg<const c10::optional<at::Scalar> &>("correction", "None")
                          .arg<bool>("keepdim", "False")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          at::OptionalIntArrayRef,
                          const c10::optional<at::Scalar> &, bool)>(var)));

            m.def(utils::FunctionSchemaBuilder("std").add_overload("MaskedPair")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .arg<bool>("unbiased", "True")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          bool)>(std)));
            m.def(utils::FunctionSchemaBuilder("std").add_overload("Tensor")
                          .arg<const at::Tensor &>("self")
                          .arg<bool>("unbiased", "True")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          bool)>(std)));

            m.def(utils::FunctionSchemaBuilder("std").add_overload("MaskedPair").add_overload("dim")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .arg<at::OptionalIntArrayRef>("dim")
                          .arg<bool>("unbiased", "True")
                          .arg<bool>("keepdim", "False")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          at::OptionalIntArrayRef,
                          bool, bool)>(std)));
            m.def(utils::FunctionSchemaBuilder("std").add_overload("Tensor").add_overload("dim")
                          .arg<const at::Tensor &>("self")
                          .arg<at::OptionalIntArrayRef>("dim")
                          .arg<bool>("unbiased", "True")
                          .arg<bool>("keepdim", "False")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          at::OptionalIntArrayRef,
                          bool, bool)>(std)));

            m.def(utils::FunctionSchemaBuilder("std").add_overload("MaskedPair").add_overload("correction")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .arg<at::OptionalIntArrayRef>("dim", "None")
                          .arg<const c10::optional<at::Scalar> &>("correction", "None")
                          .arg<bool>("keepdim", "False")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          at::OptionalIntArrayRef,
                          const c10::optional<at::Scalar> &, bool)>(std)));
            m.def(utils::FunctionSchemaBuilder("std").add_overload("Tensor").add_overload("correction")
                          .arg<const at::Tensor &>("self")
                          .arg<at::OptionalIntArrayRef>("dim", "None")
                          .arg<const c10::optional<at::Scalar> &>("correction", "None")
                          .arg<bool>("keepdim", "False")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          at::OptionalIntArrayRef,
                          const c10::optional<at::Scalar> &, bool)>(std)));

            m.def(utils::FunctionSchemaBuilder("norm").overload("MaskedPair")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .arg<const c10::optional<at::Scalar> &>("p", "None")
                          .arg<at::IntArrayRef>("dim", "[]")
                          .arg<bool>("keepdim", "False")
                          .vararg().arg<c10::optional<at::ScalarType>>("dtype", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          const c10::optional<at::Scalar> &,
                          at::IntArrayRef,
                          bool,
                          c10::optional<at::ScalarType>)>(norm)));
            m.def(utils::FunctionSchemaBuilder("norm").overload("Tensor")
                          .arg<const at::Tensor &>("self")
                          .arg<const c10::optional<at::Scalar> &>("p", "None")
                          .arg<at::IntArrayRef>("dim", "[]")
                          .arg<bool>("keepdim", "False")
                          .vararg().arg<c10::optional<at::ScalarType>>("dtype", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          const c10::optional<at::Scalar> &,
                          at::IntArrayRef,
                          bool,
                          c10::optional<at::ScalarType>)>(norm)));

            m.def(utils::FunctionSchemaBuilder("linalg_norm").overload("MaskedPair")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .arg<const at::Scalar &>("ord")
                          .arg<at::OptionalIntArrayRef>("dim", "None")
                          .arg<bool>("keepdim", "False")
                          .vararg().arg<c10::optional<at::ScalarType>>("dtype", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          const at::Scalar &,
                          at::OptionalIntArrayRef,
                          bool,
                          c10::optional<at::ScalarType>)>(linalg_norm)));
            m.def(utils::FunctionSchemaBuilder("linalg_norm").overload("Tensor")
                          .arg<const at::Tensor &>("self")
                          .arg<const at::Scalar &>("ord")
                          .arg<at::OptionalIntArrayRef>("dim", "None")
                          .arg<bool>("keepdim", "False")
                          .vararg().arg<c10::optional<at::ScalarType>>("dtype", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          const at::Scalar &,
                          at::OptionalIntArrayRef,
                          bool,
                          c10::optional<at::ScalarType>)>(linalg_norm)));

            m.def(utils::FunctionSchemaBuilder("linalg_norm").overload("MaskedPair_ord_str")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .arg<c10::string_view>("ord", "\"fro\"")
                          .arg<at::OptionalIntArrayRef>("dim", "None")
                          .arg<bool>("keepdim", "False")
                          .vararg().arg<c10::optional<at::ScalarType>>("dtype", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          c10::string_view,
                          at::OptionalIntArrayRef,
                          bool,
                          c10::optional<at::ScalarType>)>(linalg_norm)));
            m.def(utils::FunctionSchemaBuilder("linalg_norm").overload("Tensor_ord_str")
                          .arg<const at::Tensor &>("self")
                          .arg<c10::string_view>("ord", "\"fro\"")
                          .arg<at::OptionalIntArrayRef>("dim", "None")
                          .arg<bool>("keepdim", "False")
                          .vararg().arg<c10::optional<at::ScalarType>>("dtype", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          c10::string_view,
                          at::OptionalIntArrayRef,
                          bool,
                          c10::optional<at::ScalarType>)>(linalg_norm)));

            m.def(utils::FunctionSchemaBuilder("linalg_vector_norm").overload("MaskedPair")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .arg<const at::Scalar &>("ord", "2")
                          .arg<at::OptionalIntArrayRef, 1>("dim", "None")
                          .arg<bool>("keepdim", "False")
                          .vararg().arg<c10::optional<at::ScalarType>>("dtype", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          const at::Scalar &,
                          at::OptionalIntArrayRef,
                          bool,
                          c10::optional<at::ScalarType>)>(linalg_vector_norm)));
            m.def(utils::FunctionSchemaBuilder("linalg_vector_norm").overload("Tensor")
                          .arg<const at::Tensor &>("self")
                          .arg<const at::Scalar &>("ord", "2")
                          .arg<at::OptionalIntArrayRef, 1>("dim", "None")
                          .arg<bool>("keepdim", "False")
                          .vararg().arg<c10::optional<at::ScalarType>>("dtype", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          const at::Scalar &,
                          at::OptionalIntArrayRef,
                          bool,
                          c10::optional<at::ScalarType>)>(linalg_vector_norm)));

            m.def(utils::FunctionSchemaBuilder("linalg_matrix_norm").overload("MaskedPair")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .arg<const at::Scalar &>("ord")
                          .arg<at::IntArrayRef>("dim", "[-2, -1]")
                          .arg<bool>("keepdim", "False")
                          .vararg().arg<c10::optional<at::ScalarType>>("dtype", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          const at::Scalar &,
                          at::IntArrayRef,
                          bool,
                          c10::optional<at::ScalarType>)>(linalg_matrix_norm)));
            m.def(utils::FunctionSchemaBuilder("linalg_matrix_norm").overload("Tensor")
                          .arg<const at::Tensor &>("self")
                          .arg<const at::Scalar &>("ord")
                          .arg<at::IntArrayRef>("dim", "[-2, -1]")
                          .arg<bool>("keepdim", "False")
                          .vararg().arg<c10::optional<at::ScalarType>>("dtype", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          const at::Scalar &,
                          at::IntArrayRef,
                          bool,
                          c10::optional<at::ScalarType>)>(linalg_matrix_norm)));

            m.def(utils::FunctionSchemaBuilder("linalg_matrix_norm").overload("MaskedPair_str_ord")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .arg<c10::string_view>("ord", "\"fro\"")
                          .arg<at::IntArrayRef>("dim", "[-2, -1]")
                          .arg<bool>("keepdim", "False")
                          .vararg().arg<c10::optional<at::ScalarType>>("dtype", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          c10::string_view,
                          at::IntArrayRef,
                          bool,
                          c10::optional<at::ScalarType>)>(linalg_matrix_norm)));
            m.def(utils::FunctionSchemaBuilder("linalg_matrix_norm").overload("Tensor_str_ord")
                          .arg<const at::Tensor &>("self")
                          .arg<c10::string_view>("ord", "\"fro\"")
                          .arg<at::IntArrayRef>("dim", "[-2, -1]")
                          .arg<bool>("keepdim", "False")
                          .vararg().arg<c10::optional<at::ScalarType>>("dtype", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          c10::string_view,
                          at::IntArrayRef,
                          bool,
                          c10::optional<at::ScalarType>)>(linalg_matrix_norm)));

            // min max
            PT_REGISTER_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS(min,)
            PT_REGISTER_REDUCTION_INDICES_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(
                    min, , int64_t,)
            PT_REGISTER_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS(max,)
            PT_REGISTER_REDUCTION_INDICES_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(
                    max, , int64_t,)
            PT_REGISTER_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(
                    amin, , at::IntArrayRef, [])
            PT_REGISTER_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS(
                    amax, , at::IntArrayRef, [])

            m.def(utils::FunctionSchemaBuilder("argmin").add_overload("MaskedPair")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .arg<c10::optional<int64_t>>("dim", "None")
                          .arg<bool>("keepdim", "False")
                          .ret<at::Tensor>().schema().c_str(),
                  TORCH_FN(static_cast<at::Tensor (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>, c10::optional<int64_t>, bool)>(argmin)));
            m.def(utils::FunctionSchemaBuilder("argmin").add_overload("Tensor")
                          .arg<const at::Tensor &>("self")
                          .arg<c10::optional<int64_t>>("dim", "None")
                          .arg<bool>("keepdim", "False")
                          .ret<at::Tensor>().schema().c_str(),
                  TORCH_FN(static_cast<at::Tensor (*)(
                          const at::Tensor &, c10::optional<int64_t>, bool)>(argmin)));

            m.def(utils::FunctionSchemaBuilder("argmax").add_overload("MaskedPair")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .arg<c10::optional<int64_t>>("dim", "None")
                          .arg<bool>("keepdim", "False")
                          .ret<at::Tensor>().schema().c_str(),
                  TORCH_FN(static_cast<at::Tensor (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>, c10::optional<int64_t>, bool)>(argmax)));
            m.def(utils::FunctionSchemaBuilder("argmax").add_overload("Tensor")
                          .arg<const at::Tensor &>("self")
                          .arg<c10::optional<int64_t>>("dim", "None")
                          .arg<bool>("keepdim", "False")
                          .ret<at::Tensor>().schema().c_str(),
                  TORCH_FN(static_cast<at::Tensor (*)(
                          const at::Tensor &, c10::optional<int64_t>, bool)>(argmax)));

            PT_REGISTER_REDUCTION_INDICES_DIM_OPS_FORALL_TENSOR_OVERLOADS(
                    cummin, , int64_t,)
            PT_REGISTER_REDUCTION_INDICES_DIM_OPS_FORALL_TENSOR_OVERLOADS(
                    cummax, , int64_t,)

            // torch.nn.functional
            PT_REGISTER_REDUCTION_DIM_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    softmin, ,
                    int64_t, , c10::optional<at::ScalarType>, dtype, None)

            m.def(utils::FunctionSchemaBuilder("normalize").add_overload("MaskedPair")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .arg<const c10::optional<at::Scalar> &>("p", "2")
                          .arg<int64_t>("dim", "1")
                          .arg<double>("eps", "1e-12")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          const c10::optional<at::Scalar> &,
                          int64_t,
                          double)>(normalize)));
            m.def(utils::FunctionSchemaBuilder("normalize").add_overload("Tensor")
                          .arg<const at::Tensor &>("self")
                          .arg<const c10::optional<at::Scalar> &>("p", "2")
                          .arg<int64_t>("dim", "1")
                          .arg<double>("eps", "1e-12")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          const c10::optional<at::Scalar> &,
                          int64_t,
                          double)>(normalize)));

            // scaled arithmetics
            PT_REGISTER_REDUCTION_AND_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    sum, scaled,
                    at::OptionalIntArrayRef, ,
                    c10::optional<at::ScalarType>, dtype, None,
                    bool, scaled,)
            PT_REGISTER_REDUCTION_KEEPDIM_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    nansum, scaled,
                    at::OptionalIntArrayRef, ,
                    c10::optional<at::ScalarType>, dtype, None,
                    bool, scaled,)
            PT_REGISTER_REDUCTION_OPS_FORALL_TENSOR_OVERLOADS(
                    trace, scaled)
        }
    }
}
