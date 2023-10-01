#include <ATen/ATen.h>
#include <torch/library.h>

#include "../MaskedPair.h"
#include "utils/mask_utils.h"
#include "utils/schema_utils.h"
#include "utils/torch_library_utils.h"

// ops definition macros
#define PT_DEFINE_BINARY_OP(NAME, IMPL_OP, IMPL_OP_, SELF_T, OTHER_T) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                            \
    SELF_T self,                                                      \
    OTHER_T other) {                                                  \
    static constexpr auto op = IMPL_OP;                               \
    static constexpr auto op_ = IMPL_OP_;                             \
    return impl::bitwise_binary_impl(op, op_, self, other);           \
}

#define PT_DEFINE_BINARY_OP_WITH(NAME, IMPL_OP, IMPL_OP_, SELF_T, OTHER_T, ARG1_T, ARG1_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                    \
    SELF_T self,                                                                              \
    OTHER_T other,                                                                            \
    ARG1_T ARG1_NAME) {                                                                       \
    static constexpr auto op = IMPL_OP;                                                       \
    static constexpr auto op_ = IMPL_OP_;                                                     \
    return impl::bitwise_binary_impl(op, op_, self, other, ARG1_NAME);                        \
}

#define PT_DEFINE_BINARY_OP_WITH3(NAME, IMPL_OP, IMPL_OP_, SELF_T, OTHER_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                                                           \
    SELF_T self,                                                                                                                     \
    OTHER_T other,                                                                                                                   \
    ARG1_T ARG1_NAME,                                                                                                                \
    ARG2_T ARG2_NAME,                                                                                                                \
    ARG3_T ARG3_NAME) {                                                                                                              \
    static constexpr auto op = IMPL_OP;                                                                                              \
    static constexpr auto op_ = IMPL_OP_;                                                                                            \
    return impl::bitwise_binary_impl(op, op_, self, other, ARG1_NAME, ARG2_NAME, ARG3_NAME);                                         \
}

#define PT_DEFINE_BINARY_OP_(NAME, IMPL_OP_, SELF_T, OTHER_T) \
c10::intrusive_ptr<TensorMaskedPair> NAME##_(                 \
    SELF_T self,                                              \
    OTHER_T other) {                                          \
    static constexpr auto op_ = IMPL_OP_;                     \
    return impl::bitwise_binary__impl(op_, self, other);      \
}

#define PT_DEFINE_BINARY_OP__WITH(NAME, IMPL_OP_, SELF_T, OTHER_T, ARG1_T, ARG1_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME##_(                                         \
    SELF_T self,                                                                      \
    OTHER_T other,                                                                    \
    ARG1_T ARG1_NAME) {                                                               \
    static constexpr auto op_ = IMPL_OP_;                                             \
    return impl::bitwise_binary__impl(op_, self, other, ARG1_NAME);                   \
}

#define PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(NAME, TENSOR_OP, TENSOR_OP_) \
PT_DEFINE_BINARY_OP(NAME, TENSOR_OP, TENSOR_OP_, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DEFINE_BINARY_OP(NAME, TENSOR_OP, TENSOR_OP_, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &)                          \
PT_DEFINE_BINARY_OP(NAME, TENSOR_OP, TENSOR_OP_, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>)                          \
PT_DEFINE_BINARY_OP(NAME, TENSOR_OP, TENSOR_OP_, const at::Tensor &, const at::Tensor &)

#define PT_DEFINE_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(NAME, TENSOR_OP, TENSOR_OP_, SCALAR_OP, SCALAR_OP_) \
PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(NAME, TENSOR_OP, TENSOR_OP_)                                                 \
PT_DEFINE_BINARY_OP(NAME, SCALAR_OP, SCALAR_OP_, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Scalar &)         \
PT_DEFINE_BINARY_OP(NAME, SCALAR_OP, SCALAR_OP_, const at::Tensor &, const at::Scalar &)

#define PT_DEFINE_BINARY_OPS_FORALL_TENSOR_AND_SCALAR_OVERLOADS(NAME, TENSOR_OP, TENSOR_OP_, SCALAR_OP, SCALAR_OP_) \
PT_DEFINE_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(NAME, TENSOR_OP, TENSOR_OP_, SCALAR_OP, SCALAR_OP_)   \
PT_DEFINE_BINARY_OP(NAME, TENSOR_OP, TENSOR_OP_, const at::Scalar &, const_intrusive_ptr_arg_t<TensorMaskedPair>)   \
PT_DEFINE_BINARY_OP(NAME, TENSOR_OP, TENSOR_OP_, const at::Scalar &, const at::Tensor &)

#define PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, TENSOR_OP, TENSOR_OP_, ARG1_T, ARG1_NAME) \
PT_DEFINE_BINARY_OP_WITH(NAME, TENSOR_OP, TENSOR_OP_, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME) \
PT_DEFINE_BINARY_OP_WITH(NAME, TENSOR_OP, TENSOR_OP_, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1_T, ARG1_NAME)                          \
PT_DEFINE_BINARY_OP_WITH(NAME, TENSOR_OP, TENSOR_OP_, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME)                          \
PT_DEFINE_BINARY_OP_WITH(NAME, TENSOR_OP, TENSOR_OP_, const at::Tensor &, const at::Tensor &, ARG1_T, ARG1_NAME)

#define PT_DEFINE_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH(NAME, TENSOR_OP, TENSOR_OP_, SCALAR_OP, SCALAR_OP_, ARG1_T, ARG1_NAME) \
PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, TENSOR_OP, TENSOR_OP_, ARG1_T, ARG1_NAME)                                                 \
PT_DEFINE_BINARY_OP_WITH(NAME, SCALAR_OP, SCALAR_OP_, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Scalar &, ARG1_T, ARG1_NAME)         \
PT_DEFINE_BINARY_OP_WITH(NAME, SCALAR_OP, SCALAR_OP_, const at::Tensor &, const at::Scalar &, ARG1_T, ARG1_NAME)

#define PT_DEFINE_BINARY_OPS_FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(NAME, TENSOR_OP, TENSOR_OP_, SCALAR_OP, SCALAR_OP_, ARG1_T, ARG1_NAME) \
PT_DEFINE_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH(NAME, TENSOR_OP, TENSOR_OP_, SCALAR_OP, SCALAR_OP_, ARG1_T, ARG1_NAME)   \
PT_DEFINE_BINARY_OP_WITH(NAME, TENSOR_OP, TENSOR_OP_, const at::Scalar &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME)   \
PT_DEFINE_BINARY_OP_WITH(NAME, TENSOR_OP, TENSOR_OP_, const at::Scalar &, const at::Tensor &, ARG1_T, ARG1_NAME)

#define PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH3(NAME, TENSOR_OP, TENSOR_OP_, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME) \
PT_DEFINE_BINARY_OP_WITH3(NAME, TENSOR_OP, TENSOR_OP_, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME) \
PT_DEFINE_BINARY_OP_WITH3(NAME, TENSOR_OP, TENSOR_OP_, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME)                          \
PT_DEFINE_BINARY_OP_WITH3(NAME, TENSOR_OP, TENSOR_OP_, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME)                          \
PT_DEFINE_BINARY_OP_WITH3(NAME, TENSOR_OP, TENSOR_OP_, const at::Tensor &, const at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME)

#define PT_DEFINE_BINARY_OPS__FORALL_TENSOR_OVERLOADS(NAME, TENSOR_OP_) \
PT_DEFINE_BINARY_OP_(NAME, TENSOR_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DEFINE_BINARY_OP_(NAME, TENSOR_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &)                          \
PT_DEFINE_BINARY_OP_(NAME, TENSOR_OP_, at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>)                          \
PT_DEFINE_BINARY_OP_(NAME, TENSOR_OP_, at::Tensor &, const at::Tensor &)

#define PT_DEFINE_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(NAME, TENSOR_OP_, SCALAR_OP_) \
PT_DEFINE_BINARY_OPS__FORALL_TENSOR_OVERLOADS(NAME, TENSOR_OP_)                                      \
PT_DEFINE_BINARY_OP_(NAME, SCALAR_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Scalar &)    \
PT_DEFINE_BINARY_OP_(NAME, SCALAR_OP_, at::Tensor &, const at::Scalar &)

#define PT_DEFINE_BINARY_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(NAME, TENSOR_OP_, SCALAR_OP_) \
PT_DEFINE_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(NAME, TENSOR_OP_, SCALAR_OP_)   \
PT_DEFINE_BINARY_OP_(NAME, TENSOR_OP_, const at::Scalar &, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DEFINE_BINARY_OP_(NAME, TENSOR_OP_, const at::Scalar &, const at::Tensor &)

#define PT_DEFINE_BINARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, TENSOR_OP_, ARG1_T, ARG1_NAME) \
PT_DEFINE_BINARY_OP__WITH(NAME, TENSOR_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME) \
PT_DEFINE_BINARY_OP__WITH(NAME, TENSOR_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1_T, ARG1_NAME)                          \
PT_DEFINE_BINARY_OP__WITH(NAME, TENSOR_OP_, at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME)                          \
PT_DEFINE_BINARY_OP__WITH(NAME, TENSOR_OP_, at::Tensor &, const at::Tensor &, ARG1_T, ARG1_NAME)

#define PT_DEFINE_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH(NAME, TENSOR_OP_, SCALAR_OP_, ARG1_T, ARG1_NAME) \
PT_DEFINE_BINARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, TENSOR_OP_, ARG1_T, ARG1_NAME)                                      \
PT_DEFINE_BINARY_OP__WITH(NAME, SCALAR_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Scalar &, ARG1_T, ARG1_NAME)    \
PT_DEFINE_BINARY_OP__WITH(NAME, SCALAR_OP_, at::Tensor &, const at::Scalar &, ARG1_T, ARG1_NAME)

#define PT_DEFINE_BINARY_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(NAME, TENSOR_OP_, SCALAR_OP_, ARG1_T, ARG1_NAME) \
PT_DEFINE_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH(NAME, TENSOR_OP_, SCALAR_OP_, ARG1_T, ARG1_NAME)   \
PT_DEFINE_BINARY_OP__WITH(NAME, TENSOR_OP_, const at::Scalar &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME) \
PT_DEFINE_BINARY_OP__WITH(NAME, TENSOR_OP_, const at::Scalar &, const at::Tensor &, ARG1_T, ARG1_NAME)

#define PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(NAME, TENSOR_OP, TENSOR_OP_) \
PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(NAME, TENSOR_OP, TENSOR_OP_)                  \
PT_DEFINE_BINARY_OPS__FORALL_TENSOR_OVERLOADS(NAME, TENSOR_OP_)

#define PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(NAME, TENSOR_OP, TENSOR_OP_, SCALAR_OP, SCALAR_OP_) \
PT_DEFINE_BINARY_OPS_FORALL_TENSOR_AND_SCALAR_OVERLOADS(NAME, TENSOR_OP, TENSOR_OP_, SCALAR_OP, SCALAR_OP_)                  \
PT_DEFINE_BINARY_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(NAME, TENSOR_OP_, SCALAR_OP_)

#define PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, TENSOR_OP, TENSOR_OP_, ARG1_T, ARG1_NAME) \
PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, TENSOR_OP, TENSOR_OP_, ARG1_T, ARG1_NAME)                  \
PT_DEFINE_BINARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, TENSOR_OP_, ARG1_T, ARG1_NAME)

#define PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(NAME, TENSOR_OP, TENSOR_OP_, SCALAR_OP, SCALAR_OP_, ARG1_T, ARG1_NAME) \
PT_DEFINE_BINARY_OPS_FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(NAME, TENSOR_OP, TENSOR_OP_, SCALAR_OP, SCALAR_OP_, ARG1_T, ARG1_NAME)                  \
PT_DEFINE_BINARY_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(NAME, TENSOR_OP_, SCALAR_OP_, ARG1_T, ARG1_NAME)

// ops registration macros
#define PT_BINARY_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER_T) \
utils::FunctionSchemaBuilder(#NAME).add_overload(#OVERLOAD_NAME).add_overload(#POSTFIX).arg<SELF_T>("self").arg<OTHER_T>("other").ret<TensorMaskedPair>()

#define PT_REGISTER_BINARY_OP(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER_T) \
m.def(PT_BINARY_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER_T).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, OTHER_T)>(NAME)));

#define PT_REGISTER_BINARY_OP_WITH(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
m.def(PT_BINARY_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER_T).vararg().arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, OTHER_T, ARG1_T)>(NAME)));

#define PT_REGISTER_BINARY_OP_WITH2(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
m.def(PT_BINARY_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER_T).vararg().arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).arg<ARG2_T>(#ARG2_NAME, #ARG2_DEFAULT).schema().c_str(), \
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

#define PT_REGISTER_BINARY_OP__WITH2(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
m.def(PT_BINARY_SCHEMA_BUILDER(NAME##_, OVERLOAD_NAME, POSTFIX, SELF_T, OTHER_T).vararg().arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).arg<ARG2_T>(#ARG2_NAME, #ARG2_DEFAULT).schema().c_str(), \
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

#define PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_BINARY_OP_WITH2(NAME, MaskedPair, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_BINARY_OP_WITH2(NAME, MaskedPair_Tensor, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                   \
PT_REGISTER_BINARY_OP_WITH2(NAME, Tensor_MaskedPair, POSTFIX, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                   \
PT_REGISTER_BINARY_OP_WITH2(NAME, Tensor, POSTFIX, const at::Tensor &, const at::Tensor &, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

#define PT_REGISTER_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH2(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                          \
PT_REGISTER_BINARY_OP_WITH2(NAME, MaskedPair_Scalar, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Scalar &, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_BINARY_OP_WITH2(NAME, Tensor_Scalar, POSTFIX, const at::Tensor &, const at::Scalar &, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

#define PT_REGISTER_BINARY_OPS_FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH2(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH2(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)   \
PT_REGISTER_BINARY_OP_WITH2(NAME, Scalar_MaskedPair, POSTFIX, const at::Scalar &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_BINARY_OP_WITH2(NAME, Scalar_Tensor, POSTFIX, const at::Scalar &, const at::Tensor &, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

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

#define PT_REGISTER_BINARY_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_BINARY_OP__WITH2(NAME, MaskedPair, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_BINARY_OP__WITH2(NAME, MaskedPair_Tensor, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                   \
PT_REGISTER_BINARY_OP__WITH2(NAME, Tensor_MaskedPair, POSTFIX, at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                   \
PT_REGISTER_BINARY_OP__WITH2(NAME, Tensor, POSTFIX, at::Tensor &, const at::Tensor &, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

#define PT_REGISTER_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH2(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_BINARY_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                          \
PT_REGISTER_BINARY_OP__WITH2(NAME, MaskedPair_Scalar, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Scalar &, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_BINARY_OP__WITH2(NAME, Tensor_Scalar, POSTFIX, at::Tensor &, const at::Scalar &, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

#define PT_REGISTER_BINARY_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH2(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH2(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)   \
PT_REGISTER_BINARY_OP__WITH2(NAME, Scalar_MaskedPair, POSTFIX, const at::Scalar &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_BINARY_OP__WITH2(NAME, Scalar_Tensor, POSTFIX, const at::Scalar &, const at::Tensor &, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

#define PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(NAME, POSTFIX) \
PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX)                  \
PT_REGISTER_BINARY_OPS__FORALL_TENSOR_OVERLOADS(NAME, POSTFIX)

#define PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(NAME, POSTFIX) \
PT_REGISTER_BINARY_OPS_FORALL_TENSOR_AND_SCALAR_OVERLOADS(NAME, POSTFIX)                  \
PT_REGISTER_BINARY_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(NAME, POSTFIX)

#define PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)                  \
PT_REGISTER_BINARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)

#define PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                  \
PT_REGISTER_BINARY_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

#define PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_BINARY_OPS_FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)                  \
PT_REGISTER_BINARY_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)

#define PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH2(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_BINARY_OPS_FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH2(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                  \
PT_REGISTER_BINARY_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH2(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

namespace partialtorch {
    namespace ops {
        namespace impl {
            template<typename op_T, typename op__T, typename self_T, typename other_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> bitwise_binary_impl(
                    op_T &&op,
                    op__T &&op_,
                    const self_T &self,
                    const other_T &other,
                    Args &&... args) {
                if constexpr (std::is_same_v<c10::remove_intrusive_t<self_T>, TensorMaskedPair> ||
                              std::is_same_v<self_T, at::Tensor>) {
                    auto output_data = op.call(utils::get_data(self), utils::get_data(other), args...);
                    auto output_mask = utils::all_masks(utils::get_mask(self), utils::get_mask(other));
                    return masked_pair(output_data, output_mask);
                } else {
                    auto output_mask = utils::all_masks(utils::get_mask(self), utils::get_mask(other));
                    at::Tensor output_data;
                    if constexpr (!std::is_same_v<std::base_t<op__T>, nullptr_t>) {
                        output_data = op_.call(
                                at::empty_like(utils::get_data(other), at::MemoryFormat::Preserve).fill_(
                                        utils::get_data(self)),
                                utils::get_data(other), args...);
                    } else {
                        output_data = op.call(
                                at::empty_like(utils::get_data(other), at::MemoryFormat::Preserve).fill_(
                                        utils::get_data(self)),
                                utils::get_data(other), args...);
                    }
                    return masked_pair(output_data, output_mask);
                }
            }

            template<typename op__T, typename self_T, typename other_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> bitwise_binary__impl(
                    op__T &&op_,
                    self_T &self,
                    const other_T &other,
                    Args &&... args) {
                if constexpr (std::is_same_v<c10::remove_intrusive_t<self_T>, TensorMaskedPair> ||
                              std::is_same_v<self_T, at::Tensor>) {
                    auto self_data = utils::get_data(self);
                    auto output_data = op_.call(self_data, utils::get_data(other), args...);
                    auto output_mask = utils::all_masks_(utils::get_mask(self), utils::get_mask(other));
                    if constexpr (std::is_same_v<self_T, c10::intrusive_ptr<TensorMaskedPair>>) {
                        return self;
                    } else {
                        return masked_pair(output_data, output_mask);
                    }
                } else {
                    auto output_mask = utils::all_masks(utils::get_mask(self), utils::get_mask(other));
                    auto output_data = op_.call(
                            at::empty_like(utils::get_data(other), at::MemoryFormat::Preserve).fill_(
                                    utils::get_data(self)),
                            utils::get_data(other), args...);
                    return masked_pair(output_data, output_mask);
                }
            }
        }

        // logical
        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                bitwise_and, at::_ops::bitwise_and_Tensor(), at::_ops::bitwise_and__Tensor(),
                at::_ops::bitwise_and_Scalar(), at::_ops::bitwise_and__Scalar())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                bitwise_or, at::_ops::bitwise_or_Tensor(), at::_ops::bitwise_or__Tensor(),
                at::_ops::bitwise_or_Scalar(), at::_ops::bitwise_or__Scalar())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                bitwise_xor, at::_ops::bitwise_xor_Tensor(), at::_ops::bitwise_xor__Tensor(),
                at::_ops::bitwise_xor_Scalar(), at::_ops::bitwise_xor__Scalar())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                bitwise_left_shift, at::_ops::bitwise_left_shift_Tensor(), at::_ops::bitwise_left_shift__Tensor(),
                at::_ops::bitwise_left_shift_Tensor_Scalar(), at::_ops::bitwise_left_shift__Tensor_Scalar())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                bitwise_right_shift, at::_ops::bitwise_right_shift_Tensor(), at::_ops::bitwise_right_shift__Tensor(),
                at::_ops::bitwise_right_shift_Tensor_Scalar(), at::_ops::bitwise_right_shift__Tensor_Scalar())

        // python logical magic methods
        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(
                __and__, at::_ops::__and___Tensor(), nullptr, at::_ops::__and___Scalar(), nullptr)

        PT_DEFINE_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(
                __iand_, at::_ops::__iand___Tensor(), at::_ops::__iand___Scalar())

        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(
                __or__, at::_ops::__or___Tensor(), nullptr, at::_ops::__or___Scalar(), nullptr)

        PT_DEFINE_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(
                __ior_, at::_ops::__ior___Tensor(), at::_ops::__ior___Scalar())

        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(
                __xor__, at::_ops::__xor___Tensor(), nullptr, at::_ops::__xor___Scalar(), nullptr)

        PT_DEFINE_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(
                __ixor_, at::_ops::__ixor___Tensor(), at::_ops::__ixor___Scalar())

        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(
                __lshift__, at::_ops::__lshift___Tensor(), nullptr, at::_ops::__lshift___Scalar(), nullptr)

        PT_DEFINE_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(
                __ilshift_, at::_ops::__ilshift___Tensor(), at::_ops::__ilshift___Scalar())

        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(
                __rshift__, at::_ops::__rshift___Tensor(), nullptr, at::_ops::__rshift___Scalar(), nullptr)

        PT_DEFINE_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(
                __irshift_, at::_ops::__irshift___Tensor(), at::_ops::__irshift___Scalar())

        // arithmetics
        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(
                add, at::_ops::add_Tensor(), at::_ops::add__Tensor(),
                at::_ops::add_Scalar(), at::_ops::add__Scalar(),
                const at::Scalar &, alpha)

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(
                sub, at::_ops::sub_Tensor(), at::_ops::sub__Tensor(),
                at::_ops::sub_Scalar(), at::_ops::sub__Scalar(),
                const at::Scalar &, alpha)

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(
                subtract, at::_ops::subtract_Tensor(), at::_ops::subtract__Tensor(),
                at::_ops::subtract_Scalar(), at::_ops::subtract__Scalar(),
                const at::Scalar &, alpha)

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                mul, at::_ops::mul_Tensor(), at::_ops::mul__Tensor(),
                at::_ops::mul_Scalar(), at::_ops::mul__Scalar())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                multiply, at::_ops::multiply_Tensor(), at::_ops::multiply__Tensor(),
                at::_ops::multiply_Scalar(), at::_ops::multiply__Scalar())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                div, at::_ops::div_Tensor(), at::_ops::div__Tensor(),
                at::_ops::div_Scalar(), at::_ops::div__Scalar())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(
                div, at::_ops::div_Tensor_mode(), at::_ops::div__Tensor_mode(),
                at::_ops::div_Scalar_mode(), at::_ops::div__Scalar_mode(),
                c10::optional<c10::string_view>, rounding_mode)

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                divide, at::_ops::divide_Tensor(), at::_ops::divide__Tensor(),
                at::_ops::divide_Scalar(), at::_ops::divide__Scalar())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(
                divide, at::_ops::divide_Tensor_mode(), at::_ops::divide__Tensor_mode(),
                at::_ops::divide_Scalar_mode(), at::_ops::divide__Scalar_mode(),
                c10::optional<c10::string_view>, rounding_mode)

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                floor_divide, at::_ops::floor_divide(), at::_ops::floor_divide__Tensor(),
                at::_ops::floor_divide_Scalar(), at::_ops::floor_divide__Scalar())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                true_divide, at::_ops::true_divide_Tensor(), at::_ops::true_divide__Tensor(),
                at::_ops::true_divide_Scalar(), at::_ops::true_divide__Scalar())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                fmod, at::_ops::fmod_Tensor(), at::_ops::fmod__Tensor(),
                at::_ops::fmod_Scalar(), at::_ops::fmod__Scalar())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                remainder, at::_ops::remainder_Tensor(), at::_ops::remainder__Tensor(),
                at::_ops::remainder_Scalar(), at::_ops::remainder__Scalar())

        // TODO: maybe use pow_Scalar
        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                pow, at::_ops::pow_Tensor_Tensor(), at::_ops::pow__Tensor(),
                at::_ops::pow_Tensor_Scalar(), at::_ops::pow__Scalar())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                float_power, at::_ops::float_power_Tensor_Tensor(), at::_ops::float_power__Tensor(),
                at::_ops::float_power_Tensor_Scalar(), at::_ops::float_power__Scalar())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(
                atan2, at::_ops::atan2(), at::_ops::atan2_())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(
                arctan2, at::_ops::arctan2(), at::_ops::arctan2_())

        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                logaddexp, at::_ops::logaddexp(), nullptr)

        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                logaddexp2, at::_ops::logaddexp2(), nullptr)

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(
                nextafter, at::_ops::nextafter(), at::_ops::nextafter_())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(
                ldexp, at::_ops::ldexp_Tensor(), at::_ops::ldexp_())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                lerp, at::_ops::lerp_Scalar(), at::_ops::lerp__Scalar(),
                const at::Scalar &, weight)

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                lerp, at::_ops::lerp_Tensor(), at::_ops::lerp__Tensor(),
                const at::Tensor &, weight)

        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                dist, at::_ops::dist(), nullptr,
                const at::Scalar &, p)

        // comparison
        namespace impl {
            template<typename self_T, typename other_T>
            static C10_ALWAYS_INLINE bool equal_impl(
                    const self_T &self,
                    const other_T &other) {
                return at::_ops::equal::call(utils::index_non_masked(self),
                                             utils::index_non_masked(other));
            }
        }

        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                isclose, at::_ops::isclose(), nullptr,
                double, rtol, double, atol, bool, equal_nan)

        bool equal(const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                   const_intrusive_ptr_arg_t<TensorMaskedPair> other) {
            return impl::equal_impl(self, other);
        }

        bool equal(const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                   const at::Tensor &other) {
            return impl::equal_impl(self, other);
        }

        bool equal(const at::Tensor &self,
                   const_intrusive_ptr_arg_t<TensorMaskedPair> other) {
            return impl::equal_impl(self, other);
        }

        bool equal(const at::Tensor &self,
                   const at::Tensor &other) {
            return at::_ops::equal::call(self, other);
        }

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                eq, at::_ops::eq_Tensor(), at::_ops::eq__Tensor(),
                at::_ops::eq_Scalar(), at::_ops::eq__Scalar())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                ne, at::_ops::ne_Tensor(), at::_ops::ne__Tensor(),
                at::_ops::ne_Scalar(), at::_ops::ne__Scalar())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                not_equal, at::_ops::not_equal_Tensor(), at::_ops::not_equal__Tensor(),
                at::_ops::not_equal_Scalar(), at::_ops::not_equal__Scalar())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                lt, at::_ops::lt_Tensor(), at::_ops::lt__Tensor(),
                at::_ops::lt_Scalar(), at::_ops::lt__Scalar())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                less, at::_ops::less_Tensor(), at::_ops::less__Tensor(),
                at::_ops::less_Scalar(), at::_ops::less__Scalar())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                gt, at::_ops::gt_Tensor(), at::_ops::gt__Tensor(),
                at::_ops::gt_Scalar(), at::_ops::gt__Scalar())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                greater, at::_ops::greater_Tensor(), at::_ops::greater__Tensor(),
                at::_ops::greater_Scalar(), at::_ops::greater__Scalar())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                le, at::_ops::le_Tensor(), at::_ops::le__Tensor(),
                at::_ops::le_Scalar(), at::_ops::le__Scalar())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                less_equal, at::_ops::less_equal_Tensor(), at::_ops::less_equal__Tensor(),
                at::_ops::less_equal_Scalar(), at::_ops::less_equal__Scalar())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                ge, at::_ops::ge_Tensor(), at::_ops::ge__Tensor(),
                at::_ops::ge_Scalar(), at::_ops::ge__Scalar())

        PT_DEFINE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(
                greater_equal, at::_ops::greater_equal_Tensor(), at::_ops::greater_equal__Tensor(),
                at::_ops::greater_equal_Scalar(), at::_ops::greater_equal__Scalar())

        // min max
        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                min, at::_ops::min_other(), nullptr)

        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                max, at::_ops::max_other(), nullptr)

        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                minimum, at::_ops::minimum(), nullptr)

        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                maxium, at::_ops::maximum(), nullptr)

        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                fmin, at::_ops::fmin(), nullptr)

        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                fmax, at::_ops::fmax(), nullptr)

        TORCH_LIBRARY_FRAGMENT(partialtorch, m) {
            // logical
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(bitwise_and,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(bitwise_or,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(bitwise_xor,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(bitwise_left_shift,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(bitwise_right_shift,)

            // python logical magic methods
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(__and__,)
            PT_REGISTER_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(__iand_,)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(__or__,)
            PT_REGISTER_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(__ior_,)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(__xor__,)
            PT_REGISTER_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(__ixor_,)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(__lshift__,)
            PT_REGISTER_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(__ilshift_,)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(__rshift__,)
            PT_REGISTER_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(__irshift_,)

            // arithmetics
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(
                    add, ,
                    const at::Scalar &, alpha, 1)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(
                    sub, ,
                    const at::Scalar &, alpha, 1)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(
                    subtract, ,
                    const at::Scalar &, alpha, 1)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(mul,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(multiply,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(div,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(
                    div, mode,
                    c10::optional<c10::string_view>, rounding_mode, None)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(divide,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(
                    divide, mode,
                    c10::optional<c10::string_view>, rounding_mode, None)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(floor_divide,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(true_divide,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(fmod,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(remainder,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(pow,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(float_power,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(atan2,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(arctan2,)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS(logaddexp,)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS(logaddexp2,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(nextafter,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(ldexp,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                    lerp, Scalar,
                    const at::Scalar &, weight,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                    lerp, Tensor,
                    const at::Tensor &, weight,)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    dist, ,
                    const at::Scalar &, p, 2)

            // comparison
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                    isclose, , -1,
                    double, rtol, 1e-05,
                    double, atol, 1e-08,
                    bool, equal_nan, False)

            m.def(utils::FunctionSchemaBuilder("equal")
                          .overload("MaskedPair")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("other").ret<bool>().schema().c_str(),
                  TORCH_FN(static_cast<bool (*)(const_intrusive_ptr_arg_t<TensorMaskedPair>,
                                                const_intrusive_ptr_arg_t<TensorMaskedPair>)>(equal)));
            m.def(utils::FunctionSchemaBuilder("equal")
                          .overload("MaskedPair_Tensor")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .arg<const at::Tensor &>("other").ret<bool>().schema().c_str(),
                  TORCH_FN(static_cast<bool (*)(const_intrusive_ptr_arg_t<TensorMaskedPair>,
                                                const at::Tensor &)>(equal)));
            m.def(utils::FunctionSchemaBuilder("equal")
                          .overload("Tensor_MaskedPair")
                          .arg<const at::Tensor &>("self")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("other").ret<bool>().schema().c_str(),
                  TORCH_FN(static_cast<bool (*)(const at::Tensor &,
                                                const_intrusive_ptr_arg_t<TensorMaskedPair>)>(equal)));
            m.def(utils::FunctionSchemaBuilder("equal")
                          .overload("Tensor")
                          .arg<const at::Tensor &>("self")
                          .arg<const at::Tensor &>("other").ret<bool>().schema().c_str(),
                  TORCH_FN(static_cast<bool (*)(const at::Tensor &,
                                                const at::Tensor &)>(equal)));

            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(eq,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(ne,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(not_equal,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(lt,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(less,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(gt,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(greater,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(le,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(less_equal,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(ge,)
            PT_REGISTER_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(greater_equal,)

            // min max
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS(min, other)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS(max, other)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS(minimum,)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS(maxium,)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS(fmin,)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS(fmax,)
        }
    }
}
