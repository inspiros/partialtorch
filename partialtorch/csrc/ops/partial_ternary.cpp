#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <torch/library.h>

#include "../MaskedPair.h"
#include "izero_div.h"
#include "partial_bitwise_binary.h"
#include "utils/mask_utils.h"
#include "utils/fill_identity.h"
#include "utils/schema_utils.h"

// ~~~~~ ops definition macros ~~~~~
#define PT_DEFINE_TERNARY_OP(NAME, IMPL_OP, IDENTITY_OP, SELF_T, OTHER1_T, OTHER2_T)       \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                 \
    SELF_T self,                                                                           \
    OTHER1_T other1,                                                                       \
    OTHER2_T other2) {                                                                     \
    static constexpr auto op = IMPL_OP;                                                    \
    static constexpr auto fill_identity_op = IDENTITY_OP;                                  \
    return impl::partial_ternary_impl<false>(op, fill_identity_op, nullptr, self, other1, other2); \
}

#define PT_DEFINE_TERNARY_OP_WITH(NAME, IMPL_OP, IDENTITY_OP, SELF_T, OTHER1_T, OTHER2_T, ARG1_T, ARG1_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                                   \
    SELF_T self,                                                                                             \
    OTHER1_T other1,                                                                                         \
    OTHER2_T other2,                                                                                         \
    ARG1_T ARG1_NAME) {                                                                                      \
    static constexpr auto op = IMPL_OP;                                                                      \
    static constexpr auto fill_identity_op = IDENTITY_OP;                                                    \
    return impl::partial_ternary_impl<false>(op, fill_identity_op, nullptr, self, other1, other2, ARG1_NAME);\
}

#define PT_DEFINE_TERNARY_OP_WITH2(NAME, IMPL_OP, IDENTITY_OP, SELF_T, OTHER1_T, OTHER2_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                                                       \
    SELF_T self,                                                                                                                 \
    OTHER1_T other1,                                                                                                             \
    OTHER2_T other2,                                                                                                             \
    ARG1_T ARG1_NAME,                                                                                                            \
    ARG2_T ARG2_NAME) {                                                                                                          \
    static constexpr auto op = IMPL_OP;                                                                                          \
    static constexpr auto fill_identity_op = IDENTITY_OP;                                                                        \
    return impl::partial_ternary_impl<false>(op, fill_identity_op, nullptr, self, other1, other2, ARG1_NAME, ARG2_NAME);         \
}

#define PT_DEFINE_TERNARY_OP_(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, SELF_T, OTHER1_T, OTHER2_T)       \
c10::intrusive_ptr<TensorMaskedPair> NAME##_(                                                              \
    SELF_T self,                                                                                           \
    OTHER1_T other1,                                                                                       \
    OTHER2_T other2) {                                                                                     \
    static constexpr auto op_ = IMPL_OP_;                                                                  \
    static constexpr auto fill_identity_op = IDENTITY_OP;                                                  \
    static constexpr auto fill_identity_op_ = IDENTITY_OP_;                                                \
    return impl::partial_ternary__impl<false>(op_, fill_identity_op, fill_identity_op_, nullptr, self, other1, other2); \
}

#define PT_DEFINE_TERNARY_OP__WITH(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, SELF_T, OTHER1_T, OTHER2_T, ARG1_T, ARG1_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME##_(                                                                                \
    SELF_T self,                                                                                                             \
    OTHER1_T other1,                                                                                                         \
    OTHER2_T other2,                                                                                                         \
    ARG1_T ARG1_NAME) {                                                                                                      \
    static constexpr auto op_ = IMPL_OP_;                                                                                    \
    static constexpr auto fill_identity_op = IDENTITY_OP;                                                                    \
    static constexpr auto fill_identity_op_ = IDENTITY_OP_;                                                                  \
    return impl::partial_ternary__impl<false>(op_, fill_identity_op, fill_identity_op_, nullptr, self, other1, other2, ARG1_NAME); \
}

#define PT_DEFINE_TERNARY_OP__WITH2(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, SELF_T, OTHER1_T, OTHER2_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME##_(                                                                                                    \
    SELF_T self,                                                                                                                                 \
    OTHER1_T other1,                                                                                                                             \
    OTHER2_T other2,                                                                                                                             \
    ARG1_T ARG1_NAME,                                                                                                                            \
    ARG2_T ARG2_NAME) {                                                                                                                          \
    static constexpr auto op_ = IMPL_OP_;                                                                                                        \
    static constexpr auto fill_identity_op = IDENTITY_OP;                                                                                        \
    static constexpr auto fill_identity_op_ = IDENTITY_OP_;                                                                                      \
    return impl::partial_ternary__impl<false>(op_, fill_identity_op, fill_identity_op_, nullptr, self, other1, other2, ARG1_NAME, ARG2_NAME);    \
}

#define PT_DEFINE_TERNARY_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, IDENTITY_OP) \
PT_DEFINE_TERNARY_OP(NAME, IMPL_OP, IDENTITY_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DEFINE_TERNARY_OP(NAME, IMPL_OP, IDENTITY_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &)                          \
PT_DEFINE_TERNARY_OP(NAME, IMPL_OP, IDENTITY_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>)                          \
PT_DEFINE_TERNARY_OP(NAME, IMPL_OP, IDENTITY_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const at::Tensor &)                                                   \
PT_DEFINE_TERNARY_OP(NAME, IMPL_OP, IDENTITY_OP, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>)                          \
PT_DEFINE_TERNARY_OP(NAME, IMPL_OP, IDENTITY_OP, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &)                                                   \
PT_DEFINE_TERNARY_OP(NAME, IMPL_OP, IDENTITY_OP, const at::Tensor &, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>)                                                   \
PT_DEFINE_TERNARY_OP(NAME, IMPL_OP, IDENTITY_OP, const at::Tensor &, const at::Tensor &, const at::Tensor &)

#define PT_DEFINE_TERNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP, IDENTITY_OP, ARG1_T, ARG1_NAME) \
PT_DEFINE_TERNARY_OP_WITH(NAME, IMPL_OP, IDENTITY_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME) \
PT_DEFINE_TERNARY_OP_WITH(NAME, IMPL_OP, IDENTITY_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1_T, ARG1_NAME)                          \
PT_DEFINE_TERNARY_OP_WITH(NAME, IMPL_OP, IDENTITY_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME)                          \
PT_DEFINE_TERNARY_OP_WITH(NAME, IMPL_OP, IDENTITY_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const at::Tensor &, ARG1_T, ARG1_NAME)                                                   \
PT_DEFINE_TERNARY_OP_WITH(NAME, IMPL_OP, IDENTITY_OP, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME)                          \
PT_DEFINE_TERNARY_OP_WITH(NAME, IMPL_OP, IDENTITY_OP, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1_T, ARG1_NAME)                                                   \
PT_DEFINE_TERNARY_OP_WITH(NAME, IMPL_OP, IDENTITY_OP, const at::Tensor &, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME)                                                   \
PT_DEFINE_TERNARY_OP_WITH(NAME, IMPL_OP, IDENTITY_OP, const at::Tensor &, const at::Tensor &, const at::Tensor &, ARG1_T, ARG1_NAME)

#define PT_DEFINE_TERNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, IMPL_OP, IDENTITY_OP, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_TERNARY_OP_WITH2(NAME, IMPL_OP, IDENTITY_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_TERNARY_OP_WITH2(NAME, IMPL_OP, IDENTITY_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                          \
PT_DEFINE_TERNARY_OP_WITH2(NAME, IMPL_OP, IDENTITY_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                          \
PT_DEFINE_TERNARY_OP_WITH2(NAME, IMPL_OP, IDENTITY_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                                                   \
PT_DEFINE_TERNARY_OP_WITH2(NAME, IMPL_OP, IDENTITY_OP, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                          \
PT_DEFINE_TERNARY_OP_WITH2(NAME, IMPL_OP, IDENTITY_OP, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                                                   \
PT_DEFINE_TERNARY_OP_WITH2(NAME, IMPL_OP, IDENTITY_OP, const at::Tensor &, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                                                   \
PT_DEFINE_TERNARY_OP_WITH2(NAME, IMPL_OP, IDENTITY_OP, const at::Tensor &, const at::Tensor &, const at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)

#define PT_DEFINE_TERNARY_OPS__FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_) \
PT_DEFINE_TERNARY_OP_(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DEFINE_TERNARY_OP_(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &)                          \
PT_DEFINE_TERNARY_OP_(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>)                          \
PT_DEFINE_TERNARY_OP_(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const at::Tensor &)                                                   \
PT_DEFINE_TERNARY_OP_(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>)                          \
PT_DEFINE_TERNARY_OP_(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &)                                                   \
PT_DEFINE_TERNARY_OP_(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, at::Tensor &, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>)                                                   \
PT_DEFINE_TERNARY_OP_(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, at::Tensor &, const at::Tensor &, const at::Tensor &)

#define PT_DEFINE_TERNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, ARG1_T, ARG1_NAME) \
PT_DEFINE_TERNARY_OP__WITH(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME) \
PT_DEFINE_TERNARY_OP__WITH(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1_T, ARG1_NAME)                          \
PT_DEFINE_TERNARY_OP__WITH(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME)                          \
PT_DEFINE_TERNARY_OP__WITH(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const at::Tensor &, ARG1_T, ARG1_NAME)                                                   \
PT_DEFINE_TERNARY_OP__WITH(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME)                          \
PT_DEFINE_TERNARY_OP__WITH(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1_T, ARG1_NAME)                                                   \
PT_DEFINE_TERNARY_OP__WITH(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, at::Tensor &, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME)                                                   \
PT_DEFINE_TERNARY_OP__WITH(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, at::Tensor &, const at::Tensor &, const at::Tensor &, ARG1_T, ARG1_NAME)

#define PT_DEFINE_TERNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_TERNARY_OP__WITH2(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_TERNARY_OP__WITH2(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                          \
PT_DEFINE_TERNARY_OP__WITH2(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                          \
PT_DEFINE_TERNARY_OP__WITH2(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                                                   \
PT_DEFINE_TERNARY_OP__WITH2(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                          \
PT_DEFINE_TERNARY_OP__WITH2(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                                                   \
PT_DEFINE_TERNARY_OP__WITH2(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, at::Tensor &, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                                                   \
PT_DEFINE_TERNARY_OP__WITH2(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, at::Tensor &, const at::Tensor &, const at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)

#define PT_DEFINE_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_)  \
PT_DEFINE_TERNARY_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, IDENTITY_OP)                                           \
PT_DEFINE_TERNARY_OPS__FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_)

#define PT_DEFINE_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, ARG1_T, ARG1_NAME)  \
PT_DEFINE_TERNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP, IDENTITY_OP, ARG1_T, ARG1_NAME)                                           \
PT_DEFINE_TERNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, ARG1_T, ARG1_NAME)

#define PT_DEFINE_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, IMPL_OP, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)  \
PT_DEFINE_TERNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, IMPL_OP, IDENTITY_OP, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                                           \
PT_DEFINE_TERNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, IMPL_OP_, IDENTITY_OP, IDENTITY_OP_, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)

// scaled ops

// ~~~~~ ops registration macros ~~~~~
#define PT_TERNARY_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER1_T, OTHER2_T, OTHER1_NAME, OTHER2_NAME) \
utils::FunctionSchemaBuilder(#NAME).add_overload(#OVERLOAD_NAME).add_overload(#POSTFIX).arg<SELF_T>("self").arg<OTHER1_T>(#OTHER1_NAME).arg<OTHER2_T>(#OTHER2_NAME).ret<TensorMaskedPair>()

#define PT_REGISTER_TERNARY_OP(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER1_T, OTHER2_T, OTHER1_NAME, OTHER2_NAME) \
m.def(PT_TERNARY_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER1_T, OTHER2_T, OTHER1_NAME, OTHER2_NAME).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, OTHER1_T, OTHER2_T)>(NAME)));

#define PT_REGISTER_TERNARY_OP_WITH(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER1_T, OTHER2_T, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
m.def(PT_TERNARY_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER1_T, OTHER2_T, OTHER1_NAME, OTHER2_NAME).vararg(VARARG_INDEX).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, OTHER1_T, OTHER2_T, ARG1_T)>(NAME)));

#define PT_REGISTER_TERNARY_OP_WITH2(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER1_T, OTHER2_T, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
m.def(PT_TERNARY_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER1_T, OTHER2_T, OTHER1_NAME, OTHER2_NAME).vararg(VARARG_INDEX).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).arg<ARG2_T>(#ARG2_NAME, #ARG2_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, OTHER1_T, OTHER2_T, ARG1_T, ARG2_T)>(NAME)));

#define PT_REGISTER_TERNARY_OP_WITH3(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER1_T, OTHER2_T, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT) \
m.def(PT_TERNARY_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER1_T, OTHER2_T, OTHER1_NAME, OTHER2_NAME).vararg(VARARG_INDEX).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).arg<ARG2_T>(#ARG2_NAME, #ARG2_DEFAULT).arg<ARG3_T>(#ARG3_NAME, #ARG3_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, OTHER1_T, OTHER2_T, ARG1_T, ARG2_T, ARG3_T)>(NAME)));

#define PT_REGISTER_TERNARY_OP_(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER1_T, OTHER2_T, OTHER1_NAME, OTHER2_NAME) \
m.def(PT_TERNARY_SCHEMA_BUILDER(NAME##_, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER1_T, OTHER2_T, OTHER1_NAME, OTHER2_NAME).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, OTHER1_T, OTHER2_T)>(NAME##_)));

#define PT_REGISTER_TERNARY_OP__WITH(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER1_T, OTHER2_T, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
m.def(PT_TERNARY_SCHEMA_BUILDER(NAME##_, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER1_T, OTHER2_T, OTHER1_NAME, OTHER2_NAME).vararg(VARARG_INDEX).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, OTHER1_T, OTHER2_T, ARG1_T)>(NAME##_)));

#define PT_REGISTER_TERNARY_OP__WITH2(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER1_T, OTHER2_T, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
m.def(PT_TERNARY_SCHEMA_BUILDER(NAME##_, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER1_T, OTHER2_T, OTHER1_NAME, OTHER2_NAME).vararg(VARARG_INDEX).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).arg<ARG2_T>(#ARG2_NAME, #ARG2_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, OTHER1_T, OTHER2_T, ARG1_T, ARG2_T)>(NAME##_)));

#define PT_REGISTER_TERNARY_OP__WITH3(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER1_T, OTHER2_T, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT) \
m.def(PT_TERNARY_SCHEMA_BUILDER(NAME##_, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER1_T, OTHER2_T, OTHER1_NAME, OTHER2_NAME).vararg(VARARG_INDEX).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).arg<ARG2_T>(#ARG2_NAME, #ARG2_DEFAULT).arg<ARG3_T>(#ARG3_NAME, #ARG3_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, OTHER1_T, OTHER2_T, ARG1_T, ARG2_T, ARG3_T)>(NAME##_)));

#define PT_REGISTER_TERNARY_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, OTHER1_NAME, OTHER2_NAME) \
PT_REGISTER_TERNARY_OP(NAME, MP_MP_MP, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, OTHER1_NAME, OTHER2_NAME) \
PT_REGISTER_TERNARY_OP(NAME, MP_MP_Tensor, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, OTHER1_NAME, OTHER2_NAME)                      \
PT_REGISTER_TERNARY_OP(NAME, MP_Tensor_MP, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, OTHER1_NAME, OTHER2_NAME)                      \
PT_REGISTER_TERNARY_OP(NAME, MP_Tensor_Tensor, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const at::Tensor &, OTHER1_NAME, OTHER2_NAME)                                           \
PT_REGISTER_TERNARY_OP(NAME, Tensor_MP_MP, POSTFIX, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, OTHER1_NAME, OTHER2_NAME)                      \
PT_REGISTER_TERNARY_OP(NAME, Tensor_MP_Tensor, POSTFIX, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, OTHER1_NAME, OTHER2_NAME)                                           \
PT_REGISTER_TERNARY_OP(NAME, Tensor_Tensor_MP, POSTFIX, const at::Tensor &, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, OTHER1_NAME, OTHER2_NAME)                                           \
PT_REGISTER_TERNARY_OP(NAME, Tensor_Tensor_Tensor, POSTFIX, const at::Tensor &, const at::Tensor &, const at::Tensor &, OTHER1_NAME, OTHER2_NAME)

#define PT_REGISTER_TERNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_TERNARY_OP_WITH(NAME, MP_MP_MP, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_TERNARY_OP_WITH(NAME, MP_MP_Tensor, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)                      \
PT_REGISTER_TERNARY_OP_WITH(NAME, MP_Tensor_MP, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)                      \
PT_REGISTER_TERNARY_OP_WITH(NAME, MP_Tensor_Tensor, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const at::Tensor &, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)                                           \
PT_REGISTER_TERNARY_OP_WITH(NAME, Tensor_MP_MP, POSTFIX, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)                      \
PT_REGISTER_TERNARY_OP_WITH(NAME, Tensor_MP_Tensor, POSTFIX, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)                                           \
PT_REGISTER_TERNARY_OP_WITH(NAME, Tensor_Tensor_MP, POSTFIX, const at::Tensor &, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)                                           \
PT_REGISTER_TERNARY_OP_WITH(NAME, Tensor_Tensor_Tensor, POSTFIX, const at::Tensor &, const at::Tensor &, const at::Tensor &, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)

#define PT_REGISTER_TERNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_TERNARY_OP_WITH2(NAME, MP_MP_MP, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_TERNARY_OP_WITH2(NAME, MP_MP_Tensor, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                      \
PT_REGISTER_TERNARY_OP_WITH2(NAME, MP_Tensor_MP, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                      \
PT_REGISTER_TERNARY_OP_WITH2(NAME, MP_Tensor_Tensor, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const at::Tensor &, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                                           \
PT_REGISTER_TERNARY_OP_WITH2(NAME, Tensor_MP_MP, POSTFIX, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                      \
PT_REGISTER_TERNARY_OP_WITH2(NAME, Tensor_MP_Tensor, POSTFIX, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                                           \
PT_REGISTER_TERNARY_OP_WITH2(NAME, Tensor_Tensor_MP, POSTFIX, const at::Tensor &, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                                           \
PT_REGISTER_TERNARY_OP_WITH2(NAME, Tensor_Tensor_Tensor, POSTFIX, const at::Tensor &, const at::Tensor &, const at::Tensor &, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

#define PT_REGISTER_TERNARY_OPS__FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, OTHER1_NAME, OTHER2_NAME) \
PT_REGISTER_TERNARY_OP_(NAME, MP_MP_MP, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, OTHER1_NAME, OTHER2_NAME) \
PT_REGISTER_TERNARY_OP_(NAME, MP_MP_Tensor, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, OTHER1_NAME, OTHER2_NAME)                      \
PT_REGISTER_TERNARY_OP_(NAME, MP_Tensor_MP, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, OTHER1_NAME, OTHER2_NAME)                      \
PT_REGISTER_TERNARY_OP_(NAME, MP_Tensor_Tensor, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const at::Tensor &, OTHER1_NAME, OTHER2_NAME)                                           \
PT_REGISTER_TERNARY_OP_(NAME, Tensor_MP_MP, POSTFIX, at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, OTHER1_NAME, OTHER2_NAME)                      \
PT_REGISTER_TERNARY_OP_(NAME, Tensor_MP_Tensor, POSTFIX, at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, OTHER1_NAME, OTHER2_NAME)                                           \
PT_REGISTER_TERNARY_OP_(NAME, Tensor_Tensor_MP, POSTFIX, at::Tensor &, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, OTHER1_NAME, OTHER2_NAME)                                           \
PT_REGISTER_TERNARY_OP_(NAME, Tensor_Tensor_Tensor, POSTFIX, at::Tensor &, const at::Tensor &, const at::Tensor &, OTHER1_NAME, OTHER2_NAME)

#define PT_REGISTER_TERNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_TERNARY_OP__WITH(NAME, MP_MP_MP, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_TERNARY_OP__WITH(NAME, MP_MP_Tensor, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)                      \
PT_REGISTER_TERNARY_OP__WITH(NAME, MP_Tensor_MP, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)                      \
PT_REGISTER_TERNARY_OP__WITH(NAME, MP_Tensor_Tensor, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const at::Tensor &, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)                                           \
PT_REGISTER_TERNARY_OP__WITH(NAME, Tensor_MP_MP, POSTFIX, at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)                      \
PT_REGISTER_TERNARY_OP__WITH(NAME, Tensor_MP_Tensor, POSTFIX, at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)                                           \
PT_REGISTER_TERNARY_OP__WITH(NAME, Tensor_Tensor_MP, POSTFIX, at::Tensor &, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)                                           \
PT_REGISTER_TERNARY_OP__WITH(NAME, Tensor_Tensor_Tensor, POSTFIX, at::Tensor &, const at::Tensor &, const at::Tensor &, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)

#define PT_REGISTER_TERNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_TERNARY_OP__WITH2(NAME, MP_MP_MP, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_TERNARY_OP__WITH2(NAME, MP_MP_Tensor, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                      \
PT_REGISTER_TERNARY_OP__WITH2(NAME, MP_Tensor_MP, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                      \
PT_REGISTER_TERNARY_OP__WITH2(NAME, MP_Tensor_Tensor, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const at::Tensor &, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                                           \
PT_REGISTER_TERNARY_OP__WITH2(NAME, Tensor_MP_MP, POSTFIX, at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                      \
PT_REGISTER_TERNARY_OP__WITH2(NAME, Tensor_MP_Tensor, POSTFIX, at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                                           \
PT_REGISTER_TERNARY_OP__WITH2(NAME, Tensor_Tensor_MP, POSTFIX, at::Tensor &, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                                           \
PT_REGISTER_TERNARY_OP__WITH2(NAME, Tensor_Tensor_Tensor, POSTFIX, at::Tensor &, const at::Tensor &, const at::Tensor &, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

#define PT_REGISTER_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, OTHER1_NAME, OTHER2_NAME) \
PT_REGISTER_TERNARY_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, OTHER1_NAME, OTHER2_NAME)                  \
PT_REGISTER_TERNARY_OPS__FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, OTHER1_NAME, OTHER2_NAME)

#define PT_REGISTER_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_TERNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)                  \
PT_REGISTER_TERNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)

#define PT_REGISTER_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_TERNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                  \
PT_REGISTER_TERNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, OTHER1_NAME, OTHER2_NAME, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

namespace partialtorch {
    namespace ops {
        namespace impl {
            template<bool scaled = false, typename op_T, typename fill_identity_op_T, typename canonical_scale_T,
                    typename self_T, typename other1_T, typename other2_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> partial_ternary_impl(
                    op_T &&op,
                    fill_identity_op_T &&fill_identity_op,
                    canonical_scale_T &&canonical_scale,
                    const self_T &self,
                    const other1_T &other1,
                    const other2_T &other2,
                    Args &&... args) {
                if (!utils::has_tensor_mask(self) &&
                    !utils::has_tensor_mask(other1) &&
                    !utils::has_tensor_mask(other2)) {
                    return masked_pair(op.call(utils::get_data(self),
                                               utils::get_data(other1),
                                               utils::get_data(other2), args...));
                }
                auto output_data = op.call(fill_identity_op.call(self),
                                           fill_identity_op.call(other1),
                                           fill_identity_op.call(other2), args...);
                auto mask_ratio_options = output_data.scalar_type();
                at::Tensor mask_ratio;
                {
                    at::NoGradGuard g;
                    mask_ratio = op.call(
                            utils::get_tensor_mask(self, mask_ratio_options),
                            utils::get_tensor_mask(other1, mask_ratio_options),
                            utils::get_tensor_mask(other2, mask_ratio_options), args...);
                    if constexpr (scaled) {
                        if constexpr (std::is_same_v<std::base_t<canonical_scale_T>, nullptr_t>) {
                            utils::izero_ldiv_(
                                    mask_ratio,
                                    op.call(at::ones_like(utils::get_data(self)),
                                            at::ones_like(utils::get_data(other1)),
                                            at::ones_like(utils::get_data(other2)), args...)
                            );
                        } else if constexpr (std::is_same_v<std::base_t<canonical_scale_T>, at::Scalar> ||
                                             std::is_same_v<std::base_t<canonical_scale_T>, at::Tensor> ||
                                             std::is_arithmetic_v<std::base_t<canonical_scale_T>>) {
                            utils::izero_ldiv_(mask_ratio, canonical_scale);
                        } else {
                            utils::izero_ldiv_(
                                    mask_ratio,
                                    canonical_scale.call(utils::get_data(self),
                                                         utils::get_data(other1),
                                                         utils::get_data(other2), args...)
                            );
                        }
                    }
                }
                if constexpr (scaled) {
                    output_data.mul_(mask_ratio);
                }
                auto output_mask = mask_ratio.to(at::kBool);
                return masked_pair(output_data, output_mask);
            }

            template<bool scaled = false, typename op__T, typename fill_identity_op_T, typename fill_identity_op__T,
                    typename canonical_scale_T, typename self_T, typename other1_T, typename other2_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> partial_ternary__impl(
                    op__T &&op_,
                    fill_identity_op_T &&fill_identity_op,
                    fill_identity_op__T &&fill_identity_op_,
                    canonical_scale_T &&canonical_scale,
                    self_T &self,
                    const other1_T &other1,
                    const other2_T &other2,
                    Args &&... args) {
                if (!utils::has_tensor_mask(self) &&
                    !utils::has_tensor_mask(other1) &&
                    !utils::has_tensor_mask(other2)) {
                    auto self_data = utils::get_data(self);
                    return masked_pair(op_.call(self_data,
                                                utils::get_data(other1),
                                                utils::get_data(other2), args...));
                }
                auto self_data = fill_identity_op_.call(self);
                auto output_data = op_.call(self_data,
                                            fill_identity_op.call(other1),
                                            fill_identity_op.call(other2), args...);
                auto mask_ratio_options = output_data.scalar_type();
                at::Tensor mask_ratio;
                {
                    at::NoGradGuard g;
                    auto self_mask = utils::get_tensor_mask(self, mask_ratio_options);
                    mask_ratio = op_.call(
                            self_mask,
                            utils::get_tensor_mask(other1, mask_ratio_options),
                            utils::get_tensor_mask(other2, mask_ratio_options), args...);
                    if constexpr (scaled) {
                        if constexpr (std::is_same_v<std::base_t<canonical_scale_T>, nullptr_t>) {
                            utils::izero_ldiv_(
                                    mask_ratio,
                                    op_.call(at::ones_like(utils::get_data(self)),
                                             at::ones_like(utils::get_data(other1)),
                                             at::ones_like(utils::get_data(other2)), args...)
                            );
                        } else if constexpr (std::is_same_v<std::base_t<canonical_scale_T>, at::Scalar> ||
                                             std::is_same_v<std::base_t<canonical_scale_T>, at::Tensor> ||
                                             std::is_arithmetic_v<std::base_t<canonical_scale_T>>) {
                            utils::izero_ldiv_(mask_ratio, canonical_scale);
                        } else {
                            utils::izero_ldiv_(
                                    mask_ratio,
                                    canonical_scale.call(utils::get_data(self),
                                                         utils::get_data(other1),
                                                         utils::get_data(other2), args...)
                            );
                        }
                    }
                }
                if constexpr (scaled) {
                    output_data.mul_(mask_ratio);
                }
                auto output_mask = mask_ratio.to(at::kBool);
                if constexpr (std::is_same_v<c10::base_t<self_T>, TensorMaskedPair>) {
                    utils::set_mask_(self, output_mask);
                    return self;
                } else {
                    return masked_pair(output_data, output_mask);
                }
            }
        }

        PT_DEFINE_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                partial_addmm, at::_ops::addmm(), at::_ops::addmm_(),
                utils::_ops::fill_identity_zeros(), utils::_ops::fill_identity_zeros_(),
                const at::Scalar &, beta, const at::Scalar &, alpha)

        PT_DEFINE_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                partial_addbmm, at::_ops::addbmm(), at::_ops::addbmm_(),
                utils::_ops::fill_identity_zeros(), utils::_ops::fill_identity_zeros_(),
                const at::Scalar &, beta, const at::Scalar &, alpha)

        PT_DEFINE_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                partial_baddbmm, at::_ops::baddbmm(), at::_ops::baddbmm_(),
                utils::_ops::fill_identity_zeros(), utils::_ops::fill_identity_zeros_(),
                const at::Scalar &, beta, const at::Scalar &, alpha)

        PT_DEFINE_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                partial_addmv, at::_ops::addmv(), at::_ops::addmv_(),
                utils::_ops::fill_identity_zeros(), utils::_ops::fill_identity_zeros_(),
                const at::Scalar &, beta, const at::Scalar &, alpha)

        PT_DEFINE_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                partial_addr, at::_ops::addr(), at::_ops::addr_(),
                utils::_ops::fill_identity_zeros(), utils::_ops::fill_identity_zeros_(),
                const at::Scalar &, beta, const at::Scalar &, alpha)

        // bitwise
        namespace impl {
            template<typename self_T, typename other1_T, typename other2_T>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> addcmul_impl(
                    const self_T &self,
                    const other1_T &other1,
                    const other2_T &other2,
                    const at::Scalar &value) {
                return partial_add(self, partial_mul(other1, other2), value);
            }

            template<typename self_T, typename other1_T, typename other2_T>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> addcmul__impl(
                    self_T &self,
                    const other1_T &other1,
                    const other2_T &other2,
                    const at::Scalar &value) {
                return partial_add_(self, partial_mul(other1, other2), value);
            }

            template<typename self_T, typename other1_T, typename other2_T>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> addcdiv_impl(
                    const self_T &self,
                    const other1_T &other1,
                    const other2_T &other2,
                    const at::Scalar &value) {
                return partial_add(self, partial_div(other1, other2), value);
            }

            template<typename self_T, typename other1_T, typename other2_T>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> addcdiv__impl(
                    self_T &self,
                    const other1_T &other1,
                    const other2_T &other2,
                    const at::Scalar &value) {
                return partial_add_(self, partial_div(other1, other2), value);
            }
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcmul(const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                        const_intrusive_ptr_arg_t<TensorMaskedPair> tensor1,
                        const_intrusive_ptr_arg_t<TensorMaskedPair> tensor2,
                        const at::Scalar &value) {
            return impl::addcmul_impl(self, tensor1, tensor2, value);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcmul(const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                        const_intrusive_ptr_arg_t<TensorMaskedPair> tensor1,
                        const at::Tensor &tensor2,
                        const at::Scalar &value) {
            return impl::addcmul_impl(self, tensor1, tensor2, value);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcmul(const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                        const at::Tensor &tensor1,
                        const_intrusive_ptr_arg_t<TensorMaskedPair> tensor2,
                        const at::Scalar &value) {
            return impl::addcmul_impl(self, masked_pair(tensor1), tensor2, value);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcmul(const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                        const at::Tensor &tensor1,
                        const at::Tensor &tensor2,
                        const at::Scalar &value) {
            return impl::addcmul_impl(self, masked_pair(tensor1), tensor2, value);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcmul(const at::Tensor &self,
                        const_intrusive_ptr_arg_t<TensorMaskedPair> tensor1,
                        const_intrusive_ptr_arg_t<TensorMaskedPair> tensor2,
                        const at::Scalar &value) {
            return impl::addcmul_impl(masked_pair(self), tensor1, tensor2, value);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcmul(const at::Tensor &self,
                        const_intrusive_ptr_arg_t<TensorMaskedPair> tensor1,
                        const at::Tensor &tensor2,
                        const at::Scalar &value) {
            return impl::addcmul_impl(masked_pair(self), tensor1, tensor2, value);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcmul(const at::Tensor &self,
                        const at::Tensor &tensor1,
                        const_intrusive_ptr_arg_t<TensorMaskedPair> tensor2,
                        const at::Scalar &value) {
            return impl::addcmul_impl(masked_pair(self), masked_pair(tensor1), tensor2, value);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcmul(const at::Tensor &self,
                        const at::Tensor &tensor1,
                        const at::Tensor &tensor2,
                        const at::Scalar &value) {
            return impl::addcmul_impl(masked_pair(self), masked_pair(tensor1), tensor2, value);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcmul_(intrusive_ptr_arg_t<TensorMaskedPair> self,
                         const_intrusive_ptr_arg_t<TensorMaskedPair> tensor1,
                         const_intrusive_ptr_arg_t<TensorMaskedPair> tensor2,
                         const at::Scalar &value) {
            return impl::addcmul__impl(self, tensor1, tensor2, value);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcmul_(intrusive_ptr_arg_t<TensorMaskedPair> self,
                         const_intrusive_ptr_arg_t<TensorMaskedPair> tensor1,
                         const at::Tensor &tensor2,
                         const at::Scalar &value) {
            return impl::addcmul__impl(self, tensor1, tensor2, value);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcmul_(intrusive_ptr_arg_t<TensorMaskedPair> self,
                         const at::Tensor &tensor1,
                         const_intrusive_ptr_arg_t<TensorMaskedPair> tensor2,
                         const at::Scalar &value) {
            return impl::addcmul__impl(self, masked_pair(tensor1), tensor2, value);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcmul_(intrusive_ptr_arg_t<TensorMaskedPair> self,
                         const at::Tensor &tensor1,
                         const at::Tensor &tensor2,
                         const at::Scalar &value) {
            return impl::addcmul__impl(self, masked_pair(tensor1), tensor2, value);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcmul_(at::Tensor &self,
                         const_intrusive_ptr_arg_t<TensorMaskedPair> tensor1,
                         const_intrusive_ptr_arg_t<TensorMaskedPair> tensor2,
                         const at::Scalar &value) {
            auto self_ = masked_pair(self);
            return impl::addcmul__impl(self_, tensor1, tensor2, value);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcmul_(at::Tensor &self,
                         const_intrusive_ptr_arg_t<TensorMaskedPair> tensor1,
                         const at::Tensor &tensor2,
                         const at::Scalar &value) {
            auto self_ = masked_pair(self);
            return impl::addcmul__impl(self_, tensor1, tensor2, value);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcmul_(at::Tensor &self,
                         const at::Tensor &tensor1,
                         const_intrusive_ptr_arg_t<TensorMaskedPair> tensor2,
                         const at::Scalar &value) {
            auto self_ = masked_pair(self);
            return impl::addcmul__impl(self_, masked_pair(tensor1), tensor2, value);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcmul_(at::Tensor &self,
                         const at::Tensor &tensor1,
                         const at::Tensor &tensor2,
                         const at::Scalar &value) {
            auto self_ = masked_pair(self);
            return impl::addcmul__impl(self_, masked_pair(tensor1), tensor2, value);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcdiv(const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                        const_intrusive_ptr_arg_t<TensorMaskedPair> tensor1,
                        const_intrusive_ptr_arg_t<TensorMaskedPair> tensor2,
                        const at::Scalar &value) {
            return impl::addcdiv_impl(self, tensor1, tensor2, value);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcdiv(const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                        const_intrusive_ptr_arg_t<TensorMaskedPair> tensor1,
                        const at::Tensor &tensor2,
                        const at::Scalar &value) {
            return impl::addcdiv_impl(self, tensor1, tensor2, value);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcdiv(const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                        const at::Tensor &tensor1,
                        const_intrusive_ptr_arg_t<TensorMaskedPair> tensor2,
                        const at::Scalar &value) {
            return impl::addcdiv_impl(self, masked_pair(tensor1), tensor2, value);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcdiv(const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                        const at::Tensor &tensor1,
                        const at::Tensor &tensor2,
                        const at::Scalar &value) {
            return impl::addcdiv_impl(self, masked_pair(tensor1), tensor2, value);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcdiv(const at::Tensor &self,
                        const_intrusive_ptr_arg_t<TensorMaskedPair> tensor1,
                        const_intrusive_ptr_arg_t<TensorMaskedPair> tensor2,
                        const at::Scalar &value) {
            return impl::addcdiv_impl(masked_pair(self), tensor1, tensor2, value);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcdiv(const at::Tensor &self,
                        const_intrusive_ptr_arg_t<TensorMaskedPair> tensor1,
                        const at::Tensor &tensor2,
                        const at::Scalar &value) {
            return impl::addcdiv_impl(masked_pair(self), tensor1, tensor2, value);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcdiv(const at::Tensor &self,
                        const at::Tensor &tensor1,
                        const_intrusive_ptr_arg_t<TensorMaskedPair> tensor2,
                        const at::Scalar &value) {
            return impl::addcdiv_impl(masked_pair(self), masked_pair(tensor1), tensor2, value);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcdiv(const at::Tensor &self,
                        const at::Tensor &tensor1,
                        const at::Tensor &tensor2,
                        const at::Scalar &value) {
            return impl::addcdiv_impl(masked_pair(self), masked_pair(tensor1), tensor2, value);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcdiv_(intrusive_ptr_arg_t<TensorMaskedPair> self,
                         const_intrusive_ptr_arg_t<TensorMaskedPair> tensor1,
                         const_intrusive_ptr_arg_t<TensorMaskedPair> tensor2,
                         const at::Scalar &value) {
            return impl::addcdiv__impl(self, tensor1, tensor2, value);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcdiv_(intrusive_ptr_arg_t<TensorMaskedPair> self,
                         const_intrusive_ptr_arg_t<TensorMaskedPair> tensor1,
                         const at::Tensor &tensor2,
                         const at::Scalar &value) {
            return impl::addcdiv__impl(self, tensor1, tensor2, value);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcdiv_(intrusive_ptr_arg_t<TensorMaskedPair> self,
                         const at::Tensor &tensor1,
                         const_intrusive_ptr_arg_t<TensorMaskedPair> tensor2,
                         const at::Scalar &value) {
            return impl::addcdiv__impl(self, masked_pair(tensor1), tensor2, value);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcdiv_(intrusive_ptr_arg_t<TensorMaskedPair> self,
                         const at::Tensor &tensor1,
                         const at::Tensor &tensor2,
                         const at::Scalar &value) {
            return impl::addcdiv__impl(self, masked_pair(tensor1), tensor2, value);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcdiv_(at::Tensor &self,
                         const_intrusive_ptr_arg_t<TensorMaskedPair> tensor1,
                         const_intrusive_ptr_arg_t<TensorMaskedPair> tensor2,
                         const at::Scalar &value) {
            auto self_ = masked_pair(self);
            return impl::addcdiv__impl(self_, tensor1, tensor2, value);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcdiv_(at::Tensor &self,
                         const_intrusive_ptr_arg_t<TensorMaskedPair> tensor1,
                         const at::Tensor &tensor2,
                         const at::Scalar &value) {
            auto self_ = masked_pair(self);
            return impl::addcdiv__impl(self_, tensor1, tensor2, value);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcdiv_(at::Tensor &self,
                         const at::Tensor &tensor1,
                         const_intrusive_ptr_arg_t<TensorMaskedPair> tensor2,
                         const at::Scalar &value) {
            auto self_ = masked_pair(self);
            return impl::addcdiv__impl(self_, masked_pair(tensor1), tensor2, value);
        }

        c10::intrusive_ptr<TensorMaskedPair>
        partial_addcdiv_(at::Tensor &self,
                         const at::Tensor &tensor1,
                         const at::Tensor &tensor2,
                         const at::Scalar &value) {
            auto self_ = masked_pair(self);
            return impl::addcdiv__impl(self_, masked_pair(tensor1), tensor2, value);
        }

        TORCH_LIBRARY_FRAGMENT(partialtorch, m) {
            PT_REGISTER_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                    partial_addmm, , mat1, mat2, ,
                    const at::Scalar &, beta, 1,
                    const at::Scalar &, alpha, 1)
            PT_REGISTER_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                    partial_addbmm, , batch1, batch2, ,
                    const at::Scalar &, beta, 1,
                    const at::Scalar &, alpha, 1)
            PT_REGISTER_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                    partial_baddbmm, , batch1, batch2, ,
                    const at::Scalar &, beta, 1,
                    const at::Scalar &, alpha, 1)
            PT_REGISTER_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                    partial_addmv, , mat, vec, ,
                    const at::Scalar &, beta, 1,
                    const at::Scalar &, alpha, 1)
            PT_REGISTER_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                    partial_addr, , vec1, vec2, ,
                    const at::Scalar &, beta, 1,
                    const at::Scalar &, alpha, 1)
            PT_REGISTER_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                    partial_addcmul, , tensor1, tensor2, ,
                    const at::Scalar &, alpha, 1)
            PT_REGISTER_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                    partial_addcdiv, , tensor1, tensor2, ,
                    const at::Scalar &, alpha, 1)
        }
    }
}
