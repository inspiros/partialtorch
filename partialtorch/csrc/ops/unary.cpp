#include <torch/library.h>

#include "../MaskedPair.h"
#include "utils/mask_utils.h"
#include "utils/schema_utils.h"
#include "utils/torch_library_utils.h"

// ~~~~~ ops definition macros ~~~~~
#define PT_DEFINE_UNARY_OP(NAME, IMPL_OP, SELF_T) \
c10::intrusive_ptr<TensorMaskedPair> NAME(        \
    SELF_T self) {                                \
    static constexpr auto op = IMPL_OP;           \
    return impl::bitwise_unary_impl(op, self);    \
}

#define PT_DEFINE_UNARY_OP_WITH(NAME, IMPL_OP, SELF_T, ARG1_T, ARG1_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                \
    SELF_T self, ARG1_T ARG1_NAME) {                                      \
    static constexpr auto op = IMPL_OP;                                   \
    return impl::bitwise_unary_impl(op, self, ARG1_NAME);                 \
}

#define PT_DEFINE_UNARY_OP_WITH2(NAME, IMPL_OP, SELF_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                    \
    SELF_T self, ARG1_T ARG1_NAME, ARG2_T ARG2_NAME) {                                        \
    static constexpr auto op = IMPL_OP;                                                       \
    return impl::bitwise_unary_impl(op, self, ARG1_NAME, ARG2_NAME);                          \
}

#define PT_DEFINE_UNARY_OP_WITH3(NAME, IMPL_OP, SELF_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                                       \
    SELF_T self, ARG1_T ARG1_NAME, ARG2_T ARG2_NAME, ARG3_T ARG3_NAME) {                                         \
    static constexpr auto op = IMPL_OP;                                                                          \
    return impl::bitwise_unary_impl(op, self, ARG1_NAME, ARG2_NAME, ARG3_NAME);                                  \
}

#define PT_DEFINE_UNARY_OP_WITH4(NAME, IMPL_OP, SELF_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME, ARG4_T, ARG4_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                                                          \
    SELF_T self, ARG1_T ARG1_NAME, ARG2_T ARG2_NAME, ARG3_T ARG3_NAME, ARG4_T ARG4_NAME) {                                          \
    static constexpr auto op = IMPL_OP;                                                                                             \
    return impl::bitwise_unary_impl(op, self, ARG1_NAME, ARG2_NAME, ARG3_NAME, ARG4_NAME);                                          \
}

#define PT_DEFINE_UNARY_OP_(NAME, IMPL_OP_, SELF_T) \
c10::intrusive_ptr<TensorMaskedPair> NAME##_(       \
    SELF_T self) {                                  \
    static constexpr auto op_ = IMPL_OP_;           \
    return impl::bitwise_unary__impl(op_, self);    \
}

#define PT_DEFINE_UNARY_OP__WITH(NAME, IMPL_OP_, SELF_T, ARG1_T, ARG1_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME##_(                               \
    SELF_T self, ARG1_T ARG1_NAME) {                                        \
    static constexpr auto op_ = IMPL_OP_;                                   \
    return impl::bitwise_unary__impl(op_, self, ARG1_NAME);                 \
}

#define PT_DEFINE_UNARY_OP__WITH2(NAME, IMPL_OP_, SELF_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME##_(                                                   \
    SELF_T self, ARG1_T ARG1_NAME, ARG2_T ARG2_NAME) {                                          \
    static constexpr auto op_ = IMPL_OP_;                                                       \
    return impl::bitwise_unary__impl(op_, self, ARG1_NAME, ARG2_NAME);                          \
}

#define PT_DEFINE_UNARY_OP__WITH3(NAME, IMPL_OP_, SELF_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME##_(                                                                      \
    SELF_T self, ARG1_T ARG1_NAME, ARG2_T ARG2_NAME, ARG3_T ARG3_NAME) {                                           \
    static constexpr auto op_ = IMPL_OP_;                                                                          \
    return impl::bitwise_unary__impl(op_, self, ARG1_NAME, ARG2_NAME, ARG3_NAME);                                  \
}

#define PT_DEFINE_UNARY_OP__WITH4(NAME, IMPL_OP_, SELF_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME, ARG4_T, ARG4_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME##_(                                                                                         \
    SELF_T self, ARG1_T ARG1_NAME, ARG2_T ARG2_NAME, ARG3_T ARG3_NAME, ARG4_T ARG4_NAME) {                                            \
    static constexpr auto op_ = IMPL_OP_;                                                                                             \
    return impl::bitwise_unary__impl(op_, self, ARG1_NAME, ARG2_NAME, ARG3_NAME, ARG4_NAME);                                          \
}

#define PT_DEFINE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP) \
PT_DEFINE_UNARY_OP(NAME, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DEFINE_UNARY_OP(NAME, IMPL_OP, const at::Tensor &)

#define PT_DEFINE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP, ARG1_T, ARG1_NAME) \
PT_DEFINE_UNARY_OP_WITH(NAME, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME) \
PT_DEFINE_UNARY_OP_WITH(NAME, IMPL_OP, const at::Tensor &, ARG1_T, ARG1_NAME)

#define PT_DEFINE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, IMPL_OP, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_UNARY_OP_WITH2(NAME, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_UNARY_OP_WITH2(NAME, IMPL_OP, const at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)

#define PT_DEFINE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH3(NAME, IMPL_OP, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME) \
PT_DEFINE_UNARY_OP_WITH3(NAME, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME) \
PT_DEFINE_UNARY_OP_WITH3(NAME, IMPL_OP, const at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME)

#define PT_DEFINE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH4(NAME, IMPL_OP, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME, ARG4_T, ARG4_NAME) \
PT_DEFINE_UNARY_OP_WITH4(NAME, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME, ARG4_T, ARG4_NAME) \
PT_DEFINE_UNARY_OP_WITH4(NAME, IMPL_OP, const at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME, ARG4_T, ARG4_NAME)

#define PT_DEFINE_UNARY_OPS__FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP_) \
PT_DEFINE_UNARY_OP_(NAME, IMPL_OP_, intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DEFINE_UNARY_OP_(NAME, IMPL_OP_, at::Tensor &)

#define PT_DEFINE_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP_, ARG1_T, ARG1_NAME) \
PT_DEFINE_UNARY_OP__WITH(NAME, IMPL_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME) \
PT_DEFINE_UNARY_OP__WITH(NAME, IMPL_OP_, at::Tensor &, ARG1_T, ARG1_NAME)

#define PT_DEFINE_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, IMPL_OP_, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_UNARY_OP__WITH2(NAME, IMPL_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_UNARY_OP__WITH2(NAME, IMPL_OP_, at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)

#define PT_DEFINE_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH3(NAME, IMPL_OP_, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME) \
PT_DEFINE_UNARY_OP__WITH3(NAME, IMPL_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME) \
PT_DEFINE_UNARY_OP__WITH3(NAME, IMPL_OP_, at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME)

#define PT_DEFINE_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH4(NAME, IMPL_OP_, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME, ARG4_T, ARG4_NAME) \
PT_DEFINE_UNARY_OP__WITH4(NAME, IMPL_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME, ARG4_T, ARG4_NAME) \
PT_DEFINE_UNARY_OP__WITH4(NAME, IMPL_OP_, at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME, ARG4_T, ARG4_NAME)

#define PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, IMPL_OP_) \
PT_DEFINE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP)                            \
PT_DEFINE_UNARY_OPS__FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP_)

#define PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP, IMPL_OP_, ARG1_T, ARG1_NAME) \
PT_DEFINE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP, ARG1_T, ARG1_NAME)                            \
PT_DEFINE_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP_, ARG1_T, ARG1_NAME)

#define PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, IMPL_OP, IMPL_OP_, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, IMPL_OP, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                            \
PT_DEFINE_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, IMPL_OP_, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)

#define PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH3(NAME, IMPL_OP, IMPL_OP_, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME) \
PT_DEFINE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH3(NAME, IMPL_OP, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME)                            \
PT_DEFINE_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH3(NAME, IMPL_OP_, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME)

#define PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH4(NAME, IMPL_OP, IMPL_OP_, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME, ARG4_T, ARG4_NAME) \
PT_DEFINE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH4(NAME, IMPL_OP, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME, ARG4_T, ARG4_NAME)                            \
PT_DEFINE_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH4(NAME, IMPL_OP_, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME, ARG4_T, ARG4_NAME)

// ~~~~~ ops registration macros ~~~~~
#define PT_UNARY_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T) \
utils::FunctionSchemaBuilder(#NAME).add_overload(#OVERLOAD_NAME).add_overload(#POSTFIX).arg<SELF_T>("self").ret<TensorMaskedPair>()

#define PT_REGISTER_UNARY_OP(NAME, OVERLOAD_NAME, POSTFIX, SELF_T) \
m.def(PT_UNARY_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T)>(NAME)));

#define PT_REGISTER_UNARY_OP_WITH(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
m.def(PT_UNARY_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, ARG1_T)>(NAME)));

#define PT_REGISTER_UNARY_OP_WITH2(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
m.def(PT_UNARY_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).arg<ARG2_T>(#ARG2_NAME, #ARG2_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, ARG1_T, ARG2_T)>(NAME)));

#define PT_REGISTER_UNARY_OP_WITH3(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT) \
m.def(PT_UNARY_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).arg<ARG2_T>(#ARG2_NAME, #ARG2_DEFAULT).arg<ARG3_T>(#ARG3_NAME, #ARG3_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, ARG1_T, ARG2_T, ARG3_T)>(NAME)));

#define PT_REGISTER_UNARY_OP_WITH4(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT, ARG4_T, ARG4_NAME, ARG4_DEFAULT) \
m.def(PT_UNARY_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).arg<ARG2_T>(#ARG2_NAME, #ARG2_DEFAULT).arg<ARG3_T>(#ARG3_NAME, #ARG3_DEFAULT).arg<ARG4_T>(#ARG4_NAME, #ARG4_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, ARG1_T, ARG2_T, ARG3_T, ARG4_T)>(NAME)));

#define PT_REGISTER_UNARY_OP_(NAME, OVERLOAD_NAME, POSTFIX, SELF_T) \
m.def(PT_UNARY_SCHEMA_BUILDER(NAME##_, OVERLOAD_NAME, POSTFIX, SELF_T).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T)>(NAME##_)));

#define PT_REGISTER_UNARY_OP__WITH(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
m.def(PT_UNARY_SCHEMA_BUILDER(NAME##_, OVERLOAD_NAME, POSTFIX, SELF_T).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, ARG1_T)>(NAME##_)));

#define PT_REGISTER_UNARY_OP__WITH2(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
m.def(PT_UNARY_SCHEMA_BUILDER(NAME##_, OVERLOAD_NAME, POSTFIX, SELF_T).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).arg<ARG2_T>(#ARG2_NAME, #ARG2_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, ARG1_T, ARG2_T)>(NAME##_)));

#define PT_REGISTER_UNARY_OP__WITH3(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT) \
m.def(PT_UNARY_SCHEMA_BUILDER(NAME##_, OVERLOAD_NAME, POSTFIX, SELF_T).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).arg<ARG2_T>(#ARG2_NAME, #ARG2_DEFAULT).arg<ARG3_T>(#ARG3_NAME, #ARG3_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, ARG1_T, ARG2_T, ARG3_T)>(NAME##_)));

#define PT_REGISTER_UNARY_OP__WITH4(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT, ARG4_T, ARG4_NAME, ARG4_DEFAULT) \
m.def(PT_UNARY_SCHEMA_BUILDER(NAME##_, OVERLOAD_NAME, POSTFIX, SELF_T).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).arg<ARG2_T>(#ARG2_NAME, #ARG2_DEFAULT).arg<ARG3_T>(#ARG3_NAME, #ARG3_DEFAULT).arg<ARG4_T>(#ARG4_NAME, #ARG4_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, ARG1_T, ARG2_T, ARG3_T, ARG4_T)>(NAME##_)));

#define PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX) \
PT_REGISTER_UNARY_OP(NAME, MaskedPair, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_REGISTER_UNARY_OP(NAME, Tensor, POSTFIX, const at::Tensor &)

#define PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_UNARY_OP_WITH(NAME, MaskedPair, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_UNARY_OP_WITH(NAME, Tensor, POSTFIX, const at::Tensor &, ARG1_T, ARG1_NAME, ARG1_DEFAULT)

#define PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_UNARY_OP_WITH2(NAME, MaskedPair, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_UNARY_OP_WITH2(NAME, Tensor, POSTFIX, const at::Tensor &, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

#define PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH3(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT) \
PT_REGISTER_UNARY_OP_WITH3(NAME, MaskedPair, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT) \
PT_REGISTER_UNARY_OP_WITH3(NAME, Tensor, POSTFIX, const at::Tensor &, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT)

#define PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH4(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT, ARG4_T, ARG4_NAME, ARG4_DEFAULT) \
PT_REGISTER_UNARY_OP_WITH4(NAME, MaskedPair, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT, ARG4_T, ARG4_NAME, ARG4_DEFAULT) \
PT_REGISTER_UNARY_OP_WITH4(NAME, Tensor, POSTFIX, const at::Tensor &, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT, ARG4_T, ARG4_NAME, ARG4_DEFAULT)

#define PT_REGISTER_UNARY_OPS__FORALL_TENSOR_OVERLOADS(NAME, POSTFIX) \
PT_REGISTER_UNARY_OP_(NAME, MaskedPair, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_REGISTER_UNARY_OP_(NAME, Tensor, POSTFIX, at::Tensor &)

#define PT_REGISTER_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_UNARY_OP__WITH(NAME, MaskedPair, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_UNARY_OP__WITH(NAME, Tensor, POSTFIX, at::Tensor &, ARG1_T, ARG1_NAME, ARG1_DEFAULT)

#define PT_REGISTER_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_UNARY_OP__WITH2(NAME, MaskedPair, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_UNARY_OP__WITH2(NAME, Tensor, POSTFIX, at::Tensor &, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

#define PT_REGISTER_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH3(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT) \
PT_REGISTER_UNARY_OP__WITH3(NAME, MaskedPair, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT) \
PT_REGISTER_UNARY_OP__WITH3(NAME, Tensor, POSTFIX, at::Tensor &, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT)

#define PT_REGISTER_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH4(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT, ARG4_T, ARG4_NAME, ARG4_DEFAULT) \
PT_REGISTER_UNARY_OP__WITH4(NAME, MaskedPair, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT, ARG4_T, ARG4_NAME, ARG4_DEFAULT) \
PT_REGISTER_UNARY_OP__WITH4(NAME, Tensor, POSTFIX, at::Tensor &, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT, ARG4_T, ARG4_NAME, ARG4_DEFAULT)

#define PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(NAME, POSTFIX) \
PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX)                  \
PT_REGISTER_UNARY_OPS__FORALL_TENSOR_OVERLOADS(NAME, POSTFIX)

#define PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)                  \
PT_REGISTER_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)

#define PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                  \
PT_REGISTER_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

#define PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH3(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT) \
PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH3(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT)                  \
PT_REGISTER_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH3(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT)

#define PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH4(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT, ARG4_T, ARG4_NAME, ARG4_DEFAULT) \
PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH4(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT, ARG4_T, ARG4_NAME, ARG4_DEFAULT)                  \
PT_REGISTER_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH4(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT, ARG4_T, ARG4_NAME, ARG4_DEFAULT)

namespace partialtorch {
    namespace ops {
        namespace impl {
            template<typename op_T, typename self_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> bitwise_unary_impl(
                    op_T &&op,
                    const self_T &self,
                    Args &&... args) {
                auto output_data = op.call(utils::get_data(self), args...);
                auto output_mask = utils::get_mask<true>(self);
                return masked_pair(output_data, output_mask);
            }

            template<typename op__T, typename self_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> bitwise_unary__impl(
                    op__T &&op_,
                    self_T &self,
                    Args &&... args) {
                auto self_data = utils::get_data(self);
                auto output_data = op_.call(self_data, args...);
                auto output_mask = utils::get_mask<false>(self);
                if constexpr (std::is_same_v<self_T, c10::intrusive_ptr<TensorMaskedPair>>) {
                    return self;
                } else {
                    return masked_pair(output_data, output_mask);
                }
            }
        }

        // custom ops
        namespace impl {
            template<typename self_T, typename value_T>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> fill_masked_impl(
                    const self_T &self,
                    const value_T &value) {
                if constexpr (std::is_same_v<self_T, at::Tensor>) {
                    return masked_pair(self.clone());
                } else {
                    auto output_mask = utils::get_mask<true>(self);
                    auto output_data = output_mask.has_value() ?
                                       at::masked_fill(utils::get_data(self),
                                                       utils::get_mask(self).value().logical_not(), value)
                                                               : utils::get_data<true>(self);
                    return masked_pair(utils::get_data<true>(self), output_mask);
                }
            }

            template<typename self_T, typename value_T>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> fill_masked__impl(
                    self_T &self,
                    const value_T &value) {
                if (utils::has_tensor_mask(self)) {
                    auto self_data = utils::get_data(self);
                    self_data.masked_fill_(utils::get_mask(self).value().logical_not(), value);
                }
                return masked_pair(self);
            }

            template<typename self_T>
            static C10_ALWAYS_INLINE at::Tensor index_non_masked_impl(
                    const self_T &self) {
                auto self_data = utils::get_data(self);
                if (utils::has_tensor_mask(self))
                    return self_data.index({utils::get_mask(self).value()});
                return self_data.flatten();
            }

            template<typename self_T>
            static C10_ALWAYS_INLINE at::Tensor to_tensor_impl(
                    const self_T &self,
                    const at::Scalar &value) {
                if constexpr (std::is_same_v<self_T, at::Tensor>) {
                    return self;
                } else if constexpr (std::is_same_v<c10::base_t<self_T>, TensorMaskedPair>) {
                    return utils::has_tensor_mask(self) ?
                           at::masked_fill(utils::get_data(self),
                                           utils::get_mask(self).value().logical_not(), value)
                                                        : utils::get_data<true>(self);
                }
            }
        }

        c10::intrusive_ptr<TensorMaskedPair> identity(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self) {
            return masked_pair(self->data_, self->mask_);
        }

        c10::intrusive_ptr<TensorMaskedPair> identity(
                const at::Tensor &self) {
            return masked_pair(self);
        }

        c10::intrusive_ptr<TensorMaskedPair> fill_masked(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                const at::Scalar &value) {
            return impl::fill_masked_impl(self, value);
        }

        c10::intrusive_ptr<TensorMaskedPair> fill_masked(
                const at::Tensor &self,
                const at::Scalar &value) {
            return impl::fill_masked_impl(self, value);
        }

        c10::intrusive_ptr<TensorMaskedPair> fill_masked(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                const at::Tensor &value) {
            return impl::fill_masked_impl(self, value);
        }

        c10::intrusive_ptr<TensorMaskedPair> fill_masked(
                const at::Tensor &self,
                const at::Tensor &value) {
            return impl::fill_masked_impl(self, value);
        }

        c10::intrusive_ptr<TensorMaskedPair> fill_masked_(
                intrusive_ptr_arg_t<TensorMaskedPair> self,
                const at::Scalar &value) {
            return impl::fill_masked__impl(self, value);
        }

        c10::intrusive_ptr<TensorMaskedPair> fill_masked_(
                at::Tensor &self,
                const at::Scalar &value) {
            return impl::fill_masked__impl(self, value);
        }

        c10::intrusive_ptr<TensorMaskedPair> fill_masked_(
                intrusive_ptr_arg_t<TensorMaskedPair> self,
                const at::Tensor &value) {
            return impl::fill_masked__impl(self, value);
        }

        c10::intrusive_ptr<TensorMaskedPair> fill_masked_(
                at::Tensor &self,
                const at::Tensor &value) {
            return impl::fill_masked__impl(self, value);
        }

        at::Tensor to_tensor(const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                             const at::Scalar &value) {
            return impl::to_tensor_impl(self, value);
        }

        at::Tensor to_tensor(const at::Tensor &self,
                             const at::Scalar &value) {
            return impl::to_tensor_impl(self, value);
        }

        at::Tensor index_non_masked(const_intrusive_ptr_arg_t<TensorMaskedPair> self) {
            return impl::index_non_masked_impl(self);
        }

        at::Tensor index_non_masked(
                const at::Tensor &self) {
            return impl::index_non_masked_impl(self);
        }

        // torch ops
        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(abs, at::_ops::abs(), at::_ops::abs_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(absolute, at::_ops::absolute(), at::_ops::absolute_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(acos, at::_ops::acos(), at::_ops::acos_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(arccos, at::_ops::arccos(), at::_ops::arccos_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(acosh, at::_ops::acosh(), at::_ops::acosh_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(arccosh, at::_ops::arccosh(), at::_ops::arccosh_())

        PT_DEFINE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(alias, at::_ops::alias())

        PT_DEFINE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(angle, at::_ops::angle())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(asin, at::_ops::asin(), at::_ops::asin_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(arcsin, at::_ops::arcsin(), at::_ops::arcsin_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(asinh, at::_ops::asinh(), at::_ops::asinh_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(arcsinh, at::_ops::arcsinh(), at::_ops::arcsinh_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(atan, at::_ops::atan(), at::_ops::atan_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(arctan, at::_ops::arctan(), at::_ops::arctan_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(atanh, at::_ops::atanh(), at::_ops::atanh_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(arctanh, at::_ops::arctanh(), at::_ops::arctanh_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(
                bitwise_not, at::_ops::bitwise_not(), at::_ops::bitwise_not_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(ceil, at::_ops::ceil(), at::_ops::ceil_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                clamp, at::_ops::clamp(), at::_ops::clamp_(),
                const c10::optional<at::Scalar> &, min, const c10::optional<at::Scalar> &, max)

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                clamp, at::_ops::clamp_Tensor(), at::_ops::clamp__Tensor(),
                const c10::optional<at::Tensor> &, min, const c10::optional<at::Tensor> &, max)

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                clip, at::_ops::clip(), at::_ops::clip_(),
                const c10::optional<at::Scalar> &, min, const c10::optional<at::Scalar> &, max)

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                clip, at::_ops::clip_Tensor(), at::_ops::clip__Tensor(),
                const c10::optional<at::Tensor> &, min, const c10::optional<at::Tensor> &, max)

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(
                conj_physical, at::_ops::conj_physical(), at::_ops::conj_physical_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(cos, at::_ops::cos(), at::_ops::cos_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(cosh, at::_ops::cosh(), at::_ops::cosh_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(deg2rad, at::_ops::deg2rad(), at::_ops::deg2rad_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(rad2deg, at::_ops::rad2deg(), at::_ops::rad2deg_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                dropout, at::_ops::dropout(), at::_ops::dropout_(),
                double, p, bool, train)

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                feature_dropout, at::_ops::feature_dropout(), at::_ops::feature_dropout_(),
                double, p, bool, train)

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                alpha_dropout, at::_ops::alpha_dropout(), at::_ops::alpha_dropout_(),
                double, p, bool, train)

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                feature_alpha_dropout, at::_ops::feature_alpha_dropout(), at::_ops::feature_alpha_dropout_(),
                double, p, bool, train)

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(erf, at::_ops::erf(), at::_ops::erf_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(erfc, at::_ops::erfc(), at::_ops::erfc_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(erfinv, at::_ops::erfinv(), at::_ops::erfinv_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(exp, at::_ops::exp(), at::_ops::exp_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(exp2, at::_ops::exp2(), at::_ops::exp2_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(expm1, at::_ops::expm1(), at::_ops::expm1_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(fix, at::_ops::fix(), at::_ops::fix_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(floor, at::_ops::floor(), at::_ops::floor_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(frac, at::_ops::frac(), at::_ops::frac_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(lgamma, at::_ops::lgamma(), at::_ops::lgamma_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(digamma, at::_ops::digamma(), at::_ops::digamma_())

        namespace impl {
            namespace _ops {
                struct polygamma {
                    static at::Tensor call(const at::Tensor &self, int64_t n) {
                        return at::_ops::polygamma::call(n, self);
                    }
                };
            }

            template<typename self_T>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> polygamma_impl(
                    int64_t n,
                    const self_T &self) {
                static constexpr auto op = _ops::polygamma();
                return impl::bitwise_unary_impl(op, self, n);
            }
        }

        c10::intrusive_ptr<TensorMaskedPair> polygamma(
                int64_t n, const_intrusive_ptr_arg_t<TensorMaskedPair> self) {
            return impl::polygamma_impl(n, self);
        }

        c10::intrusive_ptr<TensorMaskedPair> polygamma(
                int64_t n, const at::Tensor &self) {
            return impl::polygamma_impl(n, self);
        }

        PT_DEFINE_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                polygamma, at::_ops::polygamma_(),
                int64_t, n)

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(log, at::_ops::log(), at::_ops::log_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(log10, at::_ops::log10(), at::_ops::log10_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(log1p, at::_ops::log1p(), at::_ops::log1p_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(log2, at::_ops::log2(), at::_ops::log2_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                logit, at::_ops::logit(), at::_ops::logit_(),
                const c10::optional<double>, eps)

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(i0, at::_ops::i0(), at::_ops::i0_())

        PT_DEFINE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(isnan, at::_ops::isnan())

        PT_DEFINE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(isreal, at::_ops::isreal())

        PT_DEFINE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(isfinite, at::_ops::isfinite())

        PT_DEFINE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(isinf, at::_ops::isinf())

        PT_DEFINE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(isposinf, at::_ops::isposinf())

        PT_DEFINE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(isneginf, at::_ops::isneginf())

        PT_DEFINE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(matrix_exp, at::_ops::matrix_exp())

        PT_DEFINE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                matrix_power, at::_ops::matrix_power(),
                int64_t, n)

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH3(
                nan_to_num, at::_ops::nan_to_num(), at::_ops::nan_to_num_(),
                c10::optional<double>, nan, c10::optional<double>, posinf, c10::optional<double>, neginf)

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(neg, at::_ops::neg(), at::_ops::neg_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(negative, at::_ops::negative(), at::_ops::negative_())

        PT_DEFINE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(positive, at::_ops::positive())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(
                reciprocal, at::_ops::reciprocal(), at::_ops::reciprocal_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(round, at::_ops::round(), at::_ops::round_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(rsqrt, at::_ops::rsqrt(), at::_ops::rsqrt_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(sign, at::_ops::sign(), at::_ops::sign_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(sgn, at::_ops::sgn(), at::_ops::sgn_())

        PT_DEFINE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(signbit, at::_ops::signbit())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(sin, at::_ops::sin(), at::_ops::sin_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(sinc, at::_ops::sinc(), at::_ops::sinc_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(sinh, at::_ops::sinh(), at::_ops::sinh_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(sqrt, at::_ops::sqrt(), at::_ops::sqrt_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(square, at::_ops::square(), at::_ops::square_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(tan, at::_ops::tan(), at::_ops::tan_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(tanh, at::_ops::tanh(), at::_ops::tanh_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                hardtanh, at::_ops::hardtanh(), at::_ops::hardtanh_(),
                const at::Scalar &, min_val,
                const at::Scalar &, max_val)

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                threshold, at::_ops::threshold(), at::_ops::threshold_(),
                const at::Scalar &, threshold,
                const at::Scalar &, value)

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(trunc, at::_ops::trunc(), at::_ops::trunc_())

        // activations
        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(relu, at::_ops::relu(), at::_ops::relu_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(relu6, at::_ops::relu6(), at::_ops::relu6_())

        PT_DEFINE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                glu, at::_ops::glu(),
                int64_t, dim)

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH3(
                elu, at::_ops::elu(), at::_ops::elu_(),
                const at::Scalar &, alpha,
                const at::Scalar &, scale,
                const at::Scalar &, input_scale)

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(selu, at::_ops::selu(), at::_ops::selu_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                celu, at::_ops::celu(), at::_ops::celu_(),
                const at::Scalar &, alpha)

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                leaky_relu, at::_ops::leaky_relu(), at::_ops::leaky_relu_(),
                const at::Scalar &, negative_slope)

        PT_DEFINE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                prelu, at::_ops::prelu(),
                const at::Tensor &, weight)

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH4(
                rrelu, at::_ops::rrelu(), at::_ops::rrelu_(),
                const at::Scalar &, lower,
                const at::Scalar &, upper,
                bool, training,
                c10::optional<at::Generator>, generator)

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                gelu, at::_ops::gelu(), at::_ops::gelu_(),
                c10::string_view, approximate)

        PT_DEFINE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                hardshrink, at::_ops::hardshrink(),
                const at::Scalar &, lambd)

        PT_DEFINE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                softshrink, at::_ops::softshrink(),
                const at::Scalar &, lambd)

        PT_DEFINE_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                softplus, at::_ops::softplus(),
                const at::Scalar &, beta,
                const at::Scalar &, threshold)

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(sigmoid, at::_ops::sigmoid(), at::_ops::sigmoid_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(
                hardsigmoid, at::_ops::hardsigmoid(), at::_ops::hardsigmoid_())

        PT_DEFINE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(
                log_sigmoid, at::_ops::log_sigmoid())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(silu, at::_ops::silu(), at::_ops::silu_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(mish, at::_ops::mish(), at::_ops::mish_())

        PT_DEFINE_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(hardswish, at::_ops::hardswish(), at::_ops::hardswish_())

        // torch.nn.functional
        namespace impl {
            template<std::size_t n, bool inplace = false, typename self_T>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> dropoutnd_impl(
                    const self_T &self,
                    double p,
                    bool training) {
                TORCH_CHECK_VALUE(0 <= p <= 1,
                                  "dropout probability has to be between 0 and 1, but got ",
                                  p)
                auto self_data = utils::get_data(self);
                auto inp_dim = utils::get_data(self).ndimension();
                if constexpr (n == 1) {
                    TORCH_CHECK(inp_dim == 2 || inp_dim == 3,
                                "dropout1d: Expected 2D or 3D input, but received a ",
                                inp_dim,
                                "D input. "
                                "Note that dropout1d exists to provide channel-wise dropout on inputs with 1 "
                                "spatial dimension, a channel dimension, and an optional batch dimension "
                                "(i.e. 2D or 3D inputs).")

                    bool is_batched = inp_dim == 3;
                    if constexpr (inplace) {
                        auto output_data = at::feature_dropout_(
                                is_batched ? self_data : self_data.unsqueeze_(0), p, training);
                        if (!is_batched)
                            output_data.squeeze_(0);
                        return masked_pair(self);
                    } else {
                        auto output_data = at::feature_dropout(
                                is_batched ? self_data : self_data.unsqueeze(0), p, training);
                        auto output_mask = utils::get_mask<true>(self);
                        return masked_pair(is_batched ? output_data : output_data.squeeze(0), output_mask);
                    }
                } else if constexpr (n == 2) {
                    if (inp_dim != 3 && inp_dim != 4) {
                        TORCH_WARN("dropout2d: Received a ",
                                   inp_dim,
                                   "-D input to dropout2d, which is deprecated "
                                   "and will result in an error in a future release. To retain the behavior "
                                   "and silence this warning, please use dropout instead. Note that dropout2d "
                                   "exists to provide channel-wise dropout on inputs with 2 spatial dimensions, "
                                   "a channel dimension, and an optional batch dimension (i.e. 3D or 4D inputs).")
                    }
                    // TODO: Properly support no-batch-dim inputs. For now, these are NOT supported.
                    if (inp_dim == 3) {
                        TORCH_WARN("dropout2d: Received a 3D input to dropout2d and assuming that channel-wise "
                                   "1D dropout behavior is desired - input is interpreted as shape (N, C, L), where C "
                                   "is the channel dim. This behavior will change in a future release to interpret the "
                                   "input as one without a batch dimension, i.e. shape (C, H, W). To maintain the 1D "
                                   "channel-wise dropout behavior, please switch to using dropout1d instead.")
                    }

                    if constexpr (inplace) {
                        at::feature_dropout_(self_data, p, training);
                        return masked_pair(self);
                    } else {
                        auto output_data = at::feature_dropout(self_data, p, training);
                        auto output_mask = utils::get_mask<true>(self);
                        return masked_pair(output_data, output_mask);
                    }
                } else if constexpr (n == 3) {
                    if (inp_dim != 4 && inp_dim != 5) {
                        TORCH_WARN("dropout3d: Received a ",
                                   inp_dim,
                                   "-D input to dropout3d, which is deprecated "
                                   "and will result in an error in a future release. To retain the behavior "
                                   "and silence this warning, please use dropout instead. Note that dropout3d "
                                   "exists to provide channel-wise dropout on inputs with 3 spatial dimensions, "
                                   "a channel dimension, and an optional batch dimension (i.e. 4D or 5D inputs).")
                    }

                    bool is_batched = inp_dim == 5;
                    if constexpr (inplace) {
                        auto output_data = at::feature_dropout_(
                                is_batched ? self_data : self_data.unsqueeze_(0), p, training);
                        if (!is_batched)
                            output_data.squeeze_(0);
                        return masked_pair(self);
                    } else {
                        auto output_data = at::feature_dropout(
                                is_batched ? self_data : self_data.unsqueeze(0), p, training);
                        auto output_mask = utils::get_mask<true>(self);
                        return masked_pair(is_batched ? output_data : output_data.squeeze(0), output_mask);
                    }
                } else {
                    static_assert(0 < n <= 3, "dropoutnd is only implemented for n = 1, 2, or 3");
                }
            }

            template<std::size_t n, typename self_T>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> dropoutnd__impl(
                    const self_T &self,
                    double p,
                    bool training) {
                return dropoutnd_impl<n, true>(self, p, training);
            }

            template<typename self_T>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> softsign_impl(
                    const self_T &self) {
                auto self_data = utils::get_data(self);
                auto output_data = at::div(self_data, self_data.abs().add_(1));
                auto output_mask = utils::get_mask<true>(self);
                return masked_pair(output_data, output_mask);
            }

            template<typename self_T>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> tanhshrink_impl(
                    const self_T &self) {
                auto self_data = utils::get_data(self);
                auto output_data = self_data - self_data.tanh();
                auto output_mask = utils::get_mask<true>(self);
                return masked_pair(output_data, output_mask);
            }
        }

        c10::intrusive_ptr<TensorMaskedPair> dropout1d(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self, double p, bool training) {
            return impl::dropoutnd_impl<1>(self, p, training);
        }

        c10::intrusive_ptr<TensorMaskedPair> dropout1d(
                const at::Tensor &self, double p, bool training) {
            return impl::dropoutnd_impl<1>(self, p, training);
        }

        c10::intrusive_ptr<TensorMaskedPair> dropout1d_(
                intrusive_ptr_arg_t<TensorMaskedPair> self, double p, bool training) {
            return impl::dropoutnd__impl<1>(self, p, training);
        }

        c10::intrusive_ptr<TensorMaskedPair> dropout1d_(
                at::Tensor &self, double p, bool training) {
            return impl::dropoutnd__impl<1>(self, p, training);
        }

        c10::intrusive_ptr<TensorMaskedPair> dropout2d(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self, double p, bool training) {
            return impl::dropoutnd_impl<2>(self, p, training);
        }

        c10::intrusive_ptr<TensorMaskedPair> dropout2d(
                const at::Tensor &self, double p, bool training) {
            return impl::dropoutnd_impl<2>(self, p, training);
        }

        c10::intrusive_ptr<TensorMaskedPair> dropout2d_(
                intrusive_ptr_arg_t<TensorMaskedPair> self, double p, bool training) {
            return impl::dropoutnd__impl<2>(self, p, training);
        }

        c10::intrusive_ptr<TensorMaskedPair> dropout2d_(
                at::Tensor &self, double p, bool training) {
            return impl::dropoutnd__impl<2>(self, p, training);
        }

        c10::intrusive_ptr<TensorMaskedPair> dropout3d(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self, double p, bool training) {
            return impl::dropoutnd_impl<3>(self, p, training);
        }

        c10::intrusive_ptr<TensorMaskedPair> dropout3d(
                const at::Tensor &self, double p, bool training) {
            return impl::dropoutnd_impl<3>(self, p, training);
        }

        c10::intrusive_ptr<TensorMaskedPair> dropout3d_(
                intrusive_ptr_arg_t<TensorMaskedPair> self, double p, bool training) {
            return impl::dropoutnd__impl<3>(self, p, training);
        }

        c10::intrusive_ptr<TensorMaskedPair> dropout3d_(
                at::Tensor &self, double p, bool training) {
            return impl::dropoutnd__impl<3>(self, p, training);
        }

        PT_DEFINE_UNARY_OPS_FORALL_TENSOR_OVERLOADS(
                logsigmoid, at::_ops::log_sigmoid())

        c10::intrusive_ptr<TensorMaskedPair> softsign(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self) {
            return impl::softsign_impl(self);
        }

        c10::intrusive_ptr<TensorMaskedPair> softsign(
                const at::Tensor &self) {
            return impl::softsign_impl(self);
        }

        c10::intrusive_ptr<TensorMaskedPair> tanhshrink(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self) {
            return impl::tanhshrink_impl(self);
        }

        c10::intrusive_ptr<TensorMaskedPair> tanhshrink(
                const at::Tensor &self) {
            return impl::tanhshrink_impl(self);
        }

        // properties
        namespace impl {
            template<typename op_T, typename self_T, typename... Args>
            static C10_ALWAYS_INLINE auto property_getter_impl(
                    op_T &&op,
                    const self_T &self,
                    Args &&... args) {
                return op.call(utils::get_data(self), args...);
            }

            template<bool return_self = true, typename op__T, typename self_T, typename... Args>
            static C10_ALWAYS_INLINE std::conditional_t<return_self,
                    c10::intrusive_ptr<TensorMaskedPair>,
                    void> property_setter__impl(
                    op__T &&op_,
                    self_T &self,
                    Args &&... args) {
                auto self_data = utils::get_data(self);
                op_.call(self_data, args...);
                if constexpr (return_self)
                    return masked_pair(self);
            }
        }

        int64_t dim(const_intrusive_ptr_arg_t<TensorMaskedPair> self) {
            return self->dim();
        }

        int64_t dim(const at::Tensor &self) {
            return self.dim();
        }

        int64_t dense_dim(const_intrusive_ptr_arg_t<TensorMaskedPair> self) {
            static constexpr auto op = at::_ops::dense_dim();
            return impl::property_getter_impl(op, self);
        }

        int64_t dense_dim(const at::Tensor &self) {
            static constexpr auto op = at::_ops::dense_dim();
            return impl::property_getter_impl(op, self);
        }

        int64_t size(const_intrusive_ptr_arg_t<TensorMaskedPair> self, int64_t dim) {
            static constexpr auto op = at::_ops::size_int();
            return impl::property_getter_impl(op, self, dim);
        }

        int64_t size(const at::Tensor &self, int64_t dim) {
            static constexpr auto op = at::_ops::size_int();
            return impl::property_getter_impl(op, self, dim);
        }

        PT_DEFINE_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(requires_grad, at::_ops::requires_grad_(),
                                                          bool, requires_grad)

        TORCH_LIBRARY_FRAGMENT(partialtorch, m) {
            // custom ops
            PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS(identity,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                    fill_masked, Scalar,
                    const at::Scalar &, value,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                    fill_masked, Tensor,
                    const at::Tensor &, value,)

            m.def(utils::FunctionSchemaBuilder("to_tensor").overload("MaskedPair")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .arg<const at::Scalar &>("value")
                          .ret<at::Tensor>().schema().c_str(),
                  TORCH_FN(static_cast<at::Tensor (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          const at::Scalar &)>(to_tensor)));
            m.def("partialtorch::to_tensor(Tensor(a) self, Scalar value) -> Tensor(a)",
                  TORCH_FN(static_cast<at::Tensor (*)(
                          const at::Tensor &,
                          const at::Scalar &)>(to_tensor)));

            m.def(utils::FunctionSchemaBuilder("index_non_masked").overload("MaskedPair")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .ret<at::Tensor>().schema().c_str(),
                  TORCH_FN(static_cast<at::Tensor (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>)>(index_non_masked)));
            m.def(utils::FunctionSchemaBuilder("index_non_masked").overload("Tensor")
                          .arg<const at::Tensor &>("self")
                          .ret<at::Tensor>().schema().c_str(),
                  TORCH_FN(static_cast<at::Tensor (*)(
                          const at::Tensor &)>(index_non_masked)));

            // torch ops
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(abs,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(absolute,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(acos,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(arccos,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(acosh,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(arccosh,)
            PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS(alias,)
            PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS(angle,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(asin,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(arcsin,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(asinh,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(arcsinh,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(atan,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(arctan,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(atanh,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(arctanh,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(bitwise_not,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(ceil,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                    clamp, ,
                    const c10::optional<at::Scalar> &, min, None,
                    const c10::optional<at::Scalar> &, max, None)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                    clamp, Tensor,
                    const c10::optional<at::Tensor> &, min, None,
                    const c10::optional<at::Tensor> &, max, None)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                    clip, ,
                    const c10::optional<at::Scalar> &, min, None,
                    const c10::optional<at::Scalar> &, max, None)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                    clip, Tensor,
                    const c10::optional<at::Tensor> &, min, None,
                    const c10::optional<at::Tensor> &, max, None)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(conj_physical,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(cos,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(cosh,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(deg2rad,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(rad2deg,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                    dropout, ,
                    double, p, ,
                    bool, train,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                    feature_dropout, ,
                    double, p, ,
                    bool, train,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                    alpha_dropout, ,
                    double, p, ,
                    bool, train,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                    feature_alpha_dropout, ,
                    double, p, ,
                    bool, train,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(erf,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(erfc,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(erfinv,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(exp,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(exp2,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(expm1,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(fix,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(floor,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(frac,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(lgamma,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(digamma,)

            m.def(utils::FunctionSchemaBuilder("polygamma").overload("MaskedPair")
                          .arg<int64_t>("n")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          int64_t, const_intrusive_ptr_arg_t<TensorMaskedPair>)>(polygamma)));
            m.def(utils::FunctionSchemaBuilder("polygamma").overload("Tensor")
                          .arg<int64_t>("n")
                          .arg<const at::Tensor &>("self")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          int64_t, const at::Tensor &)>(polygamma)));

            PT_REGISTER_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                    polygamma, ,
                    int64_t, n,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(log,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(log10,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(log1p,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(log2,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                    logit, ,
                    const c10::optional<double>, eps, None)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(i0,)
            PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS(isnan,)
            PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS(isreal,)
            PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS(isfinite,)
            PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS(isinf,)
            PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS(isposinf,)
            PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS(isneginf,)
            PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS(matrix_exp,)
            PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    matrix_power, ,
                    int64_t, n,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH3(
                    nan_to_num, ,
                    c10::optional<double>, nan, None,
                    c10::optional<double>, posinf, None,
                    c10::optional<double>, neginf, None)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(neg,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(negative,)
            PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS(positive,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(reciprocal,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(round,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(rsqrt,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(sign,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(sgn,)
            PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS(signbit,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(sin,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(sinc,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(sinh,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(sqrt,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(square,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(tan,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(tanh,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                    hardtanh, ,
                    const at::Scalar &, min_val, -1,
                    const at::Scalar &, max_val, 1)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                    threshold, ,
                    const at::Scalar &, threshold, ,
                    const at::Scalar &, value,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(trunc,)

            // activations
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(relu,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(relu6,)
            PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    glu, ,
                    int64_t, dim, -1)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH3(
                    elu, ,
                    const at::Scalar &, alpha, 1,
                    const at::Scalar &, scale, 1,
                    const at::Scalar &, input_scale, 1)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(selu,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                    celu, ,
                    const at::Scalar &, alpha, 1)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                    leaky_relu, ,
                    const at::Scalar &, negative_slope, 0.01)
            PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    prelu, ,
                    const at::Tensor &, weight,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH4(
                    rrelu, ,
                    const at::Scalar &, lower, 0.125,
                    const at::Scalar &, upper, 0.3333333333333333,
                    bool, training, False,
                    c10::optional<at::Generator>, generator, None)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                    gelu, ,
                    c10::string_view, approximate, 'none')
            PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    hardshrink, ,
                    const at::Scalar &, lambd, 0.5)
            PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    softshrink, ,
                    const at::Scalar &, lambd, 0.5)
            PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    softplus, ,
                    const at::Scalar &, beta, 1,
                    const at::Scalar &, threshold, 20)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(sigmoid,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(hardsigmoid,)
            PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS(log_sigmoid,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(silu,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(mish,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(hardswish,)

            // torch.nn.functional
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                    dropout1d, ,
                    double, p, ,
                    bool, train,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                    dropout2d, ,
                    double, p, ,
                    bool, train,)
            PT_REGISTER_UNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                    dropout3d, ,
                    double, p, ,
                    bool, train,)
            PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS(logsigmoid,)
            PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS(softsign,)
            PT_REGISTER_UNARY_OPS_FORALL_TENSOR_OVERLOADS(tanhshrink,)

            // properties
            m.def(utils::FunctionSchemaBuilder("dim").add_overload("MaskedPair")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .ret<int64_t>().schema().c_str(),
                  TORCH_FN(static_cast<int64_t (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>)>(dim)));
            m.def(utils::FunctionSchemaBuilder("dim").add_overload("Tensor")
                          .arg<const at::Tensor &>("self")
                          .ret<int64_t>().schema().c_str(),
                  TORCH_FN(static_cast<int64_t (*)(
                          const at::Tensor &)>(dim)));

            m.def(utils::FunctionSchemaBuilder("dense_dim").add_overload("MaskedPair")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .ret<int64_t>().schema().c_str(),
                  TORCH_FN(static_cast<int64_t (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>)>(dense_dim)));
            m.def(utils::FunctionSchemaBuilder("dense_dim").add_overload("Tensor")
                          .arg<const at::Tensor &>("self")
                          .ret<int64_t>().schema().c_str(),
                  TORCH_FN(static_cast<int64_t (*)(
                          const at::Tensor &)>(dense_dim)));

            m.def(utils::FunctionSchemaBuilder("size").add_overload("MaskedPair").add_overload("int")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .arg<int64_t>("dim")
                          .ret<int64_t>().schema().c_str(),
                  TORCH_FN(static_cast<int64_t (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          int64_t)>(size)));
            m.def(utils::FunctionSchemaBuilder("size").add_overload("Tensor").add_overload("int")
                          .arg<const at::Tensor &>("self")
                          .arg<int64_t>("dim")
                          .ret<int64_t>().schema().c_str(),
                  TORCH_FN(static_cast<int64_t (*)(
                          const at::Tensor &,
                          int64_t)>(size)));

            PT_REGISTER_UNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                    requires_grad, ,
                    bool, requires_grad, True)
        }
    }
}
