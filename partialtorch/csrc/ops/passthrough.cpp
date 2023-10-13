#include <ATen/ATen.h>
#include <torch/library.h>

#include "../MaskedPair.h"
#include "utils/mask_utils.h"
#include "utils/schema_utils.h"
#include "utils/torch_library_utils.h"

// ~~~~~ ops definition macros ~~~~~
// one to one
#define PT_DEFINE_ONE2ONE_PASSTHROUGH_OP(NAME, IMPL_OP, SELF_T) \
c10::intrusive_ptr<TensorMaskedPair> NAME(SELF_T self) {        \
    static constexpr auto op = IMPL_OP;                         \
    return impl::one2one_passthrough_impl(op, self);            \
}

#define PT_DEFINE_ONE2ONE_PASSTHROUGH_OP_WITH(NAME, IMPL_OP, SELF_T, ARG1_T, ARG1_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(SELF_T self, ARG1_T ARG1_NAME) {              \
    static constexpr auto op = IMPL_OP;                                                 \
    return impl::one2one_passthrough_impl(op, self, ARG1_NAME);                         \
}

#define PT_DEFINE_ONE2ONE_PASSTHROUGH_OP_WITH2(NAME, IMPL_OP, SELF_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(SELF_T self, ARG1_T ARG1_NAME, ARG2_T ARG2_NAME) {                \
    static constexpr auto op = IMPL_OP;                                                                     \
    return impl::one2one_passthrough_impl(op, self, ARG1_NAME, ARG2_NAME);                                  \
}

#define PT_DEFINE_ONE2ONE_PASSTHROUGH_OP_WITH3(NAME, IMPL_OP, SELF_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(SELF_T self, ARG1_T ARG1_NAME, ARG2_T ARG2_NAME, ARG3_T ARG3_NAME) {                 \
    static constexpr auto op = IMPL_OP;                                                                                        \
    return impl::one2one_passthrough_impl(op, self, ARG1_NAME, ARG2_NAME, ARG3_NAME);                                          \
}

#define PT_DEFINE_ONE2ONE_PASSTHROUGH_OP_(NAME, IMPL_OP, SELF_T) \
c10::intrusive_ptr<TensorMaskedPair> NAME##_(SELF_T self) {      \
    static constexpr auto op_ = IMPL_OP;                         \
    return impl::one2one_passthrough__impl(op_, std::move(self));\
}

#define PT_DEFINE_ONE2ONE_PASSTHROUGH_OP__WITH(NAME, IMPL_OP, SELF_T, ARG1_T, ARG1_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME##_(SELF_T self, ARG1_T ARG1_NAME) {            \
    static constexpr auto op_ = IMPL_OP;                                                 \
    return impl::one2one_passthrough__impl(op_, std::move(self), ARG1_NAME);             \
}

#define PT_DEFINE_ONE2ONE_PASSTHROUGH_OP__WITH2(NAME, IMPL_OP, SELF_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME##_(SELF_T self, ARG1_T ARG1_NAME, ARG2_T ARG2_NAME) {              \
    static constexpr auto op_ = IMPL_OP;                                                                     \
    return impl::one2one_passthrough__impl(op_, std::move(self), ARG1_NAME, ARG2_NAME);                      \
}

#define PT_DEFINE_ONE2ONE_PASSTHROUGH_OP__WITH3(NAME, IMPL_OP, SELF_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME##_(SELF_T self, ARG1_T ARG1_NAME, ARG2_T ARG2_NAME, ARG3_T ARG3_NAME) {               \
    static constexpr auto op_ = IMPL_OP;                                                                                        \
    return impl::one2one_passthrough__impl(op_, std::move(self), ARG1_NAME, ARG2_NAME, ARG3_NAME);                              \
}

#define PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP) \
PT_DEFINE_ONE2ONE_PASSTHROUGH_OP(NAME, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DEFINE_ONE2ONE_PASSTHROUGH_OP(NAME, IMPL_OP, const at::Tensor &)

#define PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP, ARG1_T, ARG1_NAME) \
PT_DEFINE_ONE2ONE_PASSTHROUGH_OP_WITH(NAME, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME) \
PT_DEFINE_ONE2ONE_PASSTHROUGH_OP_WITH(NAME, IMPL_OP, const at::Tensor &, ARG1_T, ARG1_NAME)

#define PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, IMPL_OP, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_ONE2ONE_PASSTHROUGH_OP_WITH2(NAME, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_ONE2ONE_PASSTHROUGH_OP_WITH2(NAME, IMPL_OP, const at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)

#define PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(NAME, IMPL_OP, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME) \
PT_DEFINE_ONE2ONE_PASSTHROUGH_OP_WITH3(NAME, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME) \
PT_DEFINE_ONE2ONE_PASSTHROUGH_OP_WITH3(NAME, IMPL_OP, const at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME)

#define PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS__FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP) \
PT_DEFINE_ONE2ONE_PASSTHROUGH_OP_(NAME, IMPL_OP, intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DEFINE_ONE2ONE_PASSTHROUGH_OP_(NAME, IMPL_OP, at::Tensor &)

#define PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP, ARG1_T, ARG1_NAME) \
PT_DEFINE_ONE2ONE_PASSTHROUGH_OP__WITH(NAME, IMPL_OP, intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME) \
PT_DEFINE_ONE2ONE_PASSTHROUGH_OP__WITH(NAME, IMPL_OP, at::Tensor &, ARG1_T, ARG1_NAME)

#define PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, IMPL_OP, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_ONE2ONE_PASSTHROUGH_OP__WITH2(NAME, IMPL_OP, intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_ONE2ONE_PASSTHROUGH_OP__WITH2(NAME, IMPL_OP, at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)

#define PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS__FORALL_TENSOR_OVERLOADS_WITH3(NAME, IMPL_OP, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME) \
PT_DEFINE_ONE2ONE_PASSTHROUGH_OP__WITH3(NAME, IMPL_OP, intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME) \
PT_DEFINE_ONE2ONE_PASSTHROUGH_OP__WITH3(NAME, IMPL_OP, at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME)

#define PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, IMPL_OP_) \
PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP)                            \
PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS__FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP_)

#define PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP, IMPL_OP_, ARG1_T, ARG1_NAME) \
PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP, ARG1_T, ARG1_NAME)                            \
PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP_, ARG1_T, ARG1_NAME)

#define PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, IMPL_OP, IMPL_OP_, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, IMPL_OP, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                            \
PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, IMPL_OP_, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)

#define PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH3(NAME, IMPL_OP, IMPL_OP_, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME) \
PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(NAME, IMPL_OP, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME)                            \
PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS__FORALL_TENSOR_OVERLOADS_WITH3(NAME, IMPL_OP_, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME, ARG3_T, ARG3_NAME)

// one to many
#define PT_DEFINE_ONE2MANY_PASSTHROUGH_OP(NAME, IMPL_OP, SELF_T) \
std::vector<c10::intrusive_ptr<TensorMaskedPair>> NAME(SELF_T self) { \
    static const auto op = IMPL_OP;                              \
    return impl::one2many_passthrough_impl(op, self);            \
}

#define PT_DEFINE_ONE2MANY_PASSTHROUGH_OP_WITH(NAME, IMPL_OP, SELF_T, ARG1_T, ARG1_NAME) \
std::vector<c10::intrusive_ptr<TensorMaskedPair>> NAME(SELF_T self, ARG1_T ARG1_NAME) {  \
    static const auto op = IMPL_OP;                                                      \
    return impl::one2many_passthrough_impl(op, self, ARG1_NAME);                         \
}

#define PT_DEFINE_ONE2MANY_PASSTHROUGH_OP_WITH2(NAME, IMPL_OP, SELF_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
std::vector<c10::intrusive_ptr<TensorMaskedPair>> NAME(SELF_T self, ARG1_T ARG1_NAME, ARG2_T ARG2_NAME) {    \
    static const auto op = IMPL_OP;                                                                          \
    return impl::one2many_passthrough_impl(op, self, ARG1_NAME, ARG2_NAME);                                  \
}

#define PT_DEFINE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP) \
PT_DEFINE_ONE2MANY_PASSTHROUGH_OP(NAME, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DEFINE_ONE2MANY_PASSTHROUGH_OP(NAME, IMPL_OP, const at::Tensor &)

#define PT_DEFINE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP, ARG1_T, ARG1_NAME) \
PT_DEFINE_ONE2MANY_PASSTHROUGH_OP_WITH(NAME, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME) \
PT_DEFINE_ONE2MANY_PASSTHROUGH_OP_WITH(NAME, IMPL_OP, const at::Tensor &, ARG1_T, ARG1_NAME)

#define PT_DEFINE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, IMPL_OP, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_ONE2MANY_PASSTHROUGH_OP_WITH2(NAME, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_ONE2MANY_PASSTHROUGH_OP_WITH2(NAME, IMPL_OP, const at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)

// many to one
#define PT_DEFINE_MANY2ONE_PASSTHROUGH_OP(NAME, IMPL_OP, SELF_T) \
c10::intrusive_ptr<TensorMaskedPair> NAME(SELF_T self) {         \
    static const auto op = IMPL_OP;                              \
    return impl::many2one_passthrough_impl(op, self);            \
}

#define PT_DEFINE_MANY2ONE_PASSTHROUGH_OP_WITH(NAME, IMPL_OP, SELF_T, ARG1_T, ARG1_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(SELF_T self, ARG1_T ARG1_NAME) {               \
    static const auto op = IMPL_OP;                                                      \
    return impl::many2one_passthrough_impl(op, self, ARG1_NAME);                         \
}

#define PT_DEFINE_MANY2ONE_PASSTHROUGH_OP_WITH2(NAME, IMPL_OP, SELF_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(SELF_T self, ARG1_T ARG1_NAME, ARG2_T ARG2_NAME) {                 \
    static const auto op = IMPL_OP;                                                                          \
    return impl::many2one_passthrough_impl(op, self, ARG1_NAME, ARG2_NAME);                                  \
}

#define PT_DEFINE_MANY2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP) \
PT_DEFINE_MANY2ONE_PASSTHROUGH_OP(NAME, IMPL_OP, const at::ArrayRef<c10::intrusive_ptr<TensorMaskedPair>>) \
PT_DEFINE_MANY2ONE_PASSTHROUGH_OP(NAME, IMPL_OP, const at::ArrayRef<at::Tensor>)

#define PT_DEFINE_MANY2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP, ARG1_T, ARG1_NAME) \
PT_DEFINE_MANY2ONE_PASSTHROUGH_OP_WITH(NAME, IMPL_OP, const at::ArrayRef<c10::intrusive_ptr<TensorMaskedPair>>, ARG1_T, ARG1_NAME) \
PT_DEFINE_MANY2ONE_PASSTHROUGH_OP_WITH(NAME, IMPL_OP, const at::ArrayRef<at::Tensor>, ARG1_T, ARG1_NAME)

#define PT_DEFINE_MANY2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, IMPL_OP, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_MANY2ONE_PASSTHROUGH_OP_WITH2(NAME, IMPL_OP, const at::ArrayRef<c10::intrusive_ptr<TensorMaskedPair>>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_MANY2ONE_PASSTHROUGH_OP_WITH2(NAME, IMPL_OP, const at::ArrayRef<at::Tensor>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)

// many to many
#define PT_DEFINE_MANY2MANY_PASSTHROUGH_OP(NAME, IMPL_OP, SELF_T) \
std::vector<c10::intrusive_ptr<TensorMaskedPair>> NAME(SELF_T self) { \
    static const auto op = IMPL_OP;                               \
    return impl::many2many_passthrough_impl(op, self);            \
}

#define PT_DEFINE_MANY2MANY_PASSTHROUGH_OP_WITH(NAME, IMPL_OP, SELF_T, ARG1_T, ARG1_NAME) \
std::vector<c10::intrusive_ptr<TensorMaskedPair>> NAME(SELF_T self, ARG1_T ARG1_NAME) {   \
    static const auto op = IMPL_OP;                                                       \
    return impl::many2many_passthrough_impl(op, self, ARG1_NAME);                         \
}

#define PT_DEFINE_MANY2MANY_PASSTHROUGH_OP_WITH2(NAME, IMPL_OP, SELF_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
std::vector<c10::intrusive_ptr<TensorMaskedPair>> NAME(SELF_T self, ARG1_T ARG1_NAME, ARG2_T ARG2_NAME) {     \
    static const auto op = IMPL_OP;                                                                           \
    return impl::many2many_passthrough_impl(op, self, ARG1_NAME, ARG2_NAME);                                  \
}

#define PT_DEFINE_MANY2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP) \
PT_DEFINE_MANY2MANY_PASSTHROUGH_OP(NAME, IMPL_OP, const at::ArrayRef<c10::intrusive_ptr<TensorMaskedPair>>) \
PT_DEFINE_MANY2MANY_PASSTHROUGH_OP(NAME, IMPL_OP, const at::ArrayRef<at::Tensor>)

#define PT_DEFINE_MANY2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP, ARG1_T, ARG1_NAME) \
PT_DEFINE_MANY2MANY_PASSTHROUGH_OP_WITH(NAME, IMPL_OP, const at::ArrayRef<c10::intrusive_ptr<TensorMaskedPair>>, ARG1_T, ARG1_NAME) \
PT_DEFINE_MANY2MANY_PASSTHROUGH_OP_WITH(NAME, IMPL_OP, const at::ArrayRef<at::Tensor>, ARG1_T, ARG1_NAME)

#define PT_DEFINE_MANY2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, IMPL_OP, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_MANY2MANY_PASSTHROUGH_OP_WITH2(NAME, IMPL_OP, const at::ArrayRef<c10::intrusive_ptr<TensorMaskedPair>>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_MANY2MANY_PASSTHROUGH_OP_WITH2(NAME, IMPL_OP, const at::ArrayRef<at::Tensor>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)

// ~~~~~ ops registration macros ~~~~~
#define PT_PASSTHROUGH_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T) \
utils::FunctionSchemaBuilder(#NAME).add_overload(#OVERLOAD_NAME).add_overload(#POSTFIX).arg<SELF_T>("self")

#define PT_ONE2ONE_PASSTHROUGH_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T) \
PT_PASSTHROUGH_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).ret<TensorMaskedPair>()

#define PT_ONE2MANY_PASSTHROUGH_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T) \
PT_PASSTHROUGH_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).ret<std::vector<TensorMaskedPair>>()

#define PT_MANY2ONE_PASSTHROUGH_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T) \
PT_PASSTHROUGH_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).ret<TensorMaskedPair>()

#define PT_MANY2MANY_PASSTHROUGH_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T) \
PT_PASSTHROUGH_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).ret<std::vector<TensorMaskedPair>>()

// one to one
#define PT_REGISTER_ONE2ONE_PASSTHROUGH_OP(NAME, OVERLOAD_NAME, POSTFIX, SELF_T) \
m.def(PT_ONE2ONE_PASSTHROUGH_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T)>(NAME)));

#define PT_REGISTER_ONE2ONE_PASSTHROUGH_OP_WITH(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
m.def(PT_ONE2ONE_PASSTHROUGH_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, ARG1_T)>(NAME)));

#define PT_REGISTER_ONE2ONE_PASSTHROUGH_OP_WITH2(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
m.def(PT_ONE2ONE_PASSTHROUGH_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).arg<ARG2_T>(#ARG2_NAME, #ARG2_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, ARG1_T, ARG2_T)>(NAME)));

#define PT_REGISTER_ONE2ONE_PASSTHROUGH_OP_WITH3(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT) \
m.def(PT_ONE2ONE_PASSTHROUGH_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).arg<ARG2_T>(#ARG2_NAME, #ARG2_DEFAULT).arg<ARG3_T>(#ARG3_NAME, #ARG3_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, ARG1_T, ARG2_T, ARG3_T)>(NAME)));

#define PT_REGISTER_ONE2ONE_PASSTHROUGH_OP_(NAME, OVERLOAD_NAME, POSTFIX, SELF_T) \
m.def(PT_ONE2ONE_PASSTHROUGH_SCHEMA_BUILDER(NAME##_, OVERLOAD_NAME, POSTFIX, SELF_T).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T)>(NAME##_)));

#define PT_REGISTER_ONE2ONE_PASSTHROUGH_OP__WITH(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
m.def(PT_ONE2ONE_PASSTHROUGH_SCHEMA_BUILDER(NAME##_, OVERLOAD_NAME, POSTFIX, SELF_T).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, ARG1_T)>(NAME##_)));

#define PT_REGISTER_ONE2ONE_PASSTHROUGH_OP__WITH2(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
m.def(PT_ONE2ONE_PASSTHROUGH_SCHEMA_BUILDER(NAME##_, OVERLOAD_NAME, POSTFIX, SELF_T).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).arg<ARG2_T>(#ARG2_NAME, #ARG2_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, ARG1_T, ARG2_T)>(NAME##_)));

#define PT_REGISTER_ONE2ONE_PASSTHROUGH_OP__WITH3(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT) \
m.def(PT_ONE2ONE_PASSTHROUGH_SCHEMA_BUILDER(NAME##_, OVERLOAD_NAME, POSTFIX, SELF_T).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).arg<ARG2_T>(#ARG2_NAME, #ARG2_DEFAULT).arg<ARG3_T>(#ARG3_NAME, #ARG3_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, ARG1_T, ARG2_T, ARG3_T)>(NAME##_)));

#define PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX) \
PT_REGISTER_ONE2ONE_PASSTHROUGH_OP(NAME, MaskedPair, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_REGISTER_ONE2ONE_PASSTHROUGH_OP(NAME, Tensor, POSTFIX, const at::Tensor &)

#define PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_ONE2ONE_PASSTHROUGH_OP_WITH(NAME, MaskedPair, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_ONE2ONE_PASSTHROUGH_OP_WITH(NAME, Tensor, POSTFIX, const at::Tensor &, ARG1_T, ARG1_NAME, ARG1_DEFAULT)

#define PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_ONE2ONE_PASSTHROUGH_OP_WITH2(NAME, MaskedPair, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_ONE2ONE_PASSTHROUGH_OP_WITH2(NAME, Tensor, POSTFIX, const at::Tensor &, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

#define PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT) \
PT_REGISTER_ONE2ONE_PASSTHROUGH_OP_WITH3(NAME, MaskedPair, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT) \
PT_REGISTER_ONE2ONE_PASSTHROUGH_OP_WITH3(NAME, Tensor, POSTFIX, const at::Tensor &, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT)

#define PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS__FORALL_TENSOR_OVERLOADS(NAME, POSTFIX) \
PT_REGISTER_ONE2ONE_PASSTHROUGH_OP_(NAME, MaskedPair, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_REGISTER_ONE2ONE_PASSTHROUGH_OP_(NAME, Tensor, POSTFIX, at::Tensor &)

#define PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_ONE2ONE_PASSTHROUGH_OP__WITH(NAME, MaskedPair, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_ONE2ONE_PASSTHROUGH_OP__WITH(NAME, Tensor, POSTFIX, at::Tensor &, ARG1_T, ARG1_NAME, ARG1_DEFAULT)

#define PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_ONE2ONE_PASSTHROUGH_OP__WITH2(NAME, MaskedPair, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_ONE2ONE_PASSTHROUGH_OP__WITH2(NAME, Tensor, POSTFIX, at::Tensor &, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

#define PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS__FORALL_TENSOR_OVERLOADS_WITH3(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT) \
PT_REGISTER_ONE2ONE_PASSTHROUGH_OP__WITH3(NAME, MaskedPair, POSTFIX, intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT) \
PT_REGISTER_ONE2ONE_PASSTHROUGH_OP__WITH3(NAME, Tensor, POSTFIX, at::Tensor &, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT)

#define PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(NAME, POSTFIX) \
PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX)                  \
PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS__FORALL_TENSOR_OVERLOADS(NAME, POSTFIX)

#define PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)                  \
PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT)

#define PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)                  \
PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

#define PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH3(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT) \
PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT)                  \
PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS__FORALL_TENSOR_OVERLOADS_WITH3(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT)

// one to many
#define PT_REGISTER_ONE2MANY_PASSTHROUGH_OP(NAME, OVERLOAD_NAME, POSTFIX, SELF_T) \
m.def(PT_ONE2MANY_PASSTHROUGH_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).schema().c_str(), \
    TORCH_FN(static_cast<std::vector<c10::intrusive_ptr<TensorMaskedPair>> (*)(SELF_T)>(NAME)));

#define PT_REGISTER_ONE2MANY_PASSTHROUGH_OP_WITH(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
m.def(PT_ONE2MANY_PASSTHROUGH_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<std::vector<c10::intrusive_ptr<TensorMaskedPair>> (*)(SELF_T, ARG1_T)>(NAME)));

#define PT_REGISTER_ONE2MANY_PASSTHROUGH_OP_WITH2(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
m.def(PT_ONE2MANY_PASSTHROUGH_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).arg<ARG2_T>(#ARG2_NAME, #ARG2_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<std::vector<c10::intrusive_ptr<TensorMaskedPair>> (*)(SELF_T, ARG1_T, ARG2_T)>(NAME)));

#define PT_REGISTER_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX) \
PT_REGISTER_ONE2MANY_PASSTHROUGH_OP(NAME, MaskedPair, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_REGISTER_ONE2MANY_PASSTHROUGH_OP(NAME, Tensor, POSTFIX, const at::Tensor &)

#define PT_REGISTER_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_ONE2MANY_PASSTHROUGH_OP_WITH(NAME, MaskedPair, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_ONE2MANY_PASSTHROUGH_OP_WITH(NAME, Tensor, POSTFIX, const at::Tensor &, ARG1_T, ARG1_NAME, ARG1_DEFAULT)

#define PT_REGISTER_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_ONE2MANY_PASSTHROUGH_OP_WITH2(NAME, MaskedPair, POSTFIX, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_ONE2MANY_PASSTHROUGH_OP_WITH2(NAME, Tensor, POSTFIX, const at::Tensor &, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

// many to one
#define PT_REGISTER_MANY2ONE_PASSTHROUGH_OP(NAME, OVERLOAD_NAME, POSTFIX, SELF_T) \
m.def(PT_MANY2ONE_PASSTHROUGH_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T)>(NAME)));

#define PT_REGISTER_MANY2ONE_PASSTHROUGH_OP_WITH(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
m.def(PT_MANY2ONE_PASSTHROUGH_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, ARG1_T)>(NAME)));

#define PT_REGISTER_MANY2ONE_PASSTHROUGH_OP_WITH2(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
m.def(PT_MANY2ONE_PASSTHROUGH_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).arg<ARG2_T>(#ARG2_NAME, #ARG2_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(SELF_T, ARG1_T, ARG2_T)>(NAME)));

#define PT_REGISTER_MANY2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX) \
PT_REGISTER_MANY2ONE_PASSTHROUGH_OP(NAME, MaskedPair, POSTFIX, const at::ArrayRef<c10::intrusive_ptr<TensorMaskedPair>>) \
PT_REGISTER_MANY2ONE_PASSTHROUGH_OP(NAME, Tensor, POSTFIX, const at::ArrayRef<at::Tensor>)

#define PT_REGISTER_MANY2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_MANY2ONE_PASSTHROUGH_OP_WITH(NAME, MaskedPair, POSTFIX, const at::ArrayRef<c10::intrusive_ptr<TensorMaskedPair>>, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_MANY2ONE_PASSTHROUGH_OP_WITH(NAME, Tensor, POSTFIX, const at::ArrayRef<at::Tensor>, ARG1_T, ARG1_NAME, ARG1_DEFAULT)

#define PT_REGISTER_MANY2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_MANY2ONE_PASSTHROUGH_OP_WITH2(NAME, MaskedPair, POSTFIX, const at::ArrayRef<c10::intrusive_ptr<TensorMaskedPair>>, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_MANY2ONE_PASSTHROUGH_OP_WITH2(NAME, Tensor, POSTFIX, const at::ArrayRef<at::Tensor>, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

// many to many
#define PT_REGISTER_MANY2MANY_PASSTHROUGH_OP(NAME, OVERLOAD_NAME, POSTFIX, SELF_T) \
m.def(PT_MANY2MANY_PASSTHROUGH_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).schema().c_str(), \
    TORCH_FN(static_cast<std::vector<c10::intrusive_ptr<TensorMaskedPair>> (*)(SELF_T)>(NAME)));

#define PT_REGISTER_MANY2MANY_PASSTHROUGH_OP_WITH(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
m.def(PT_MANY2MANY_PASSTHROUGH_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<std::vector<c10::intrusive_ptr<TensorMaskedPair>> (*)(SELF_T, ARG1_T)>(NAME)));

#define PT_REGISTER_MANY2MANY_PASSTHROUGH_OP_WITH2(NAME, OVERLOAD_NAME, POSTFIX, SELF_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
m.def(PT_MANY2MANY_PASSTHROUGH_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, SELF_T).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).arg<ARG2_T>(#ARG2_NAME, #ARG2_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<std::vector<c10::intrusive_ptr<TensorMaskedPair>> (*)(SELF_T, ARG1_T, ARG2_T)>(NAME)));

#define PT_REGISTER_MANY2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX) \
PT_REGISTER_MANY2MANY_PASSTHROUGH_OP(NAME, MaskedPair, POSTFIX, const at::ArrayRef<c10::intrusive_ptr<TensorMaskedPair>>) \
PT_REGISTER_MANY2MANY_PASSTHROUGH_OP(NAME, Tensor, POSTFIX, const at::ArrayRef<at::Tensor>)

#define PT_REGISTER_MANY2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_MANY2MANY_PASSTHROUGH_OP_WITH(NAME, MaskedPair, POSTFIX, const at::ArrayRef<c10::intrusive_ptr<TensorMaskedPair>>, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_MANY2MANY_PASSTHROUGH_OP_WITH(NAME, Tensor, POSTFIX, const at::ArrayRef<at::Tensor>, ARG1_T, ARG1_NAME, ARG1_DEFAULT)

#define PT_REGISTER_MANY2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_MANY2MANY_PASSTHROUGH_OP_WITH2(NAME, MaskedPair, POSTFIX, const at::ArrayRef<c10::intrusive_ptr<TensorMaskedPair>>, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_MANY2MANY_PASSTHROUGH_OP_WITH2(NAME, Tensor, POSTFIX, const at::ArrayRef<at::Tensor>, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

namespace partialtorch {
    namespace ops {
        namespace impl {
            template<bool transform_mask = true, typename op_T, typename self_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> one2one_passthrough_impl(
                    op_T &&op,
                    self_T self,
                    Args &&... args) {
                auto output_data = op.call(utils::get_data(self), args...);
                if constexpr (transform_mask) {
                    auto input_mask = utils::get_mask(self);
                    auto output_mask = input_mask.has_value() ? op.call(input_mask.value(), args...)
                                                              : c10::optional<at::Tensor>{};
                    return masked_pair(output_data, output_mask);
                } else {
                    return masked_pair(output_data, utils::get_mask(self));
                }
            }

            template<bool transform_mask = true, typename op__T, typename self_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> one2one_passthrough__impl(
                    op__T &&op_,
                    const self_T &self,
                    Args &&... args) {
                auto input_data = utils::get_data(self);
                op_.call(input_data, args...);
                if constexpr (transform_mask) {
                    auto input_mask = utils::get_mask(self);
                    if (input_mask.has_value())
                        op_.call(input_mask.value(), args...);
                }
                return masked_pair(self);
            }

            template<typename op_T, typename self_T, typename... Args>
            static C10_ALWAYS_INLINE std::vector<c10::intrusive_ptr<TensorMaskedPair>> one2many_passthrough_impl(
                    op_T &&op,
                    self_T self,
                    Args &&... args) {
                auto output_data = op.call(utils::get_data(self), args...);
                auto input_mask = utils::get_mask(self);
                if (input_mask.has_value()) {
                    auto output_masks = op.call(input_mask.value(), args...);
                    return utils::wrap_masked_pair(output_data, output_masks);
                }
                return utils::wrap_masked_pair(output_data, input_mask);
            }

            template<typename op_T, typename self_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> many2one_passthrough_impl(
                    op_T &&op,
                    const at::ArrayRef<self_T> &tensors,
                    Args &&... args) {
                auto output_data = op.call(utils::get_data(tensors), args...);
                auto input_masks = utils::get_mask(tensors);
                bool has_any_mask = false;
                for (const auto i: c10::irange(input_masks.size())) {
                    if (input_masks[i].has_value()) {
                        has_any_mask = true;
                        break;
                    }
                }
                if (has_any_mask) {
                    std::vector<at::Tensor> input_masks_;
                    input_masks_.reserve(input_masks.size());
                    for (const auto i: c10::irange(input_masks.size())) {
                        input_masks_.emplace_back(input_masks[i].value_or(
                                at::ones_like(utils::get_data(tensors[i]),
                                              at::TensorOptions().dtype(at::kBool))));
                    }
                    auto output_mask = op.call(input_masks_, args...);
                    return masked_pair(output_data, output_mask);
                }
                return masked_pair(output_data);
            }

            template<typename op_T, typename self_T, typename... Args>
            static C10_ALWAYS_INLINE std::vector<c10::intrusive_ptr<TensorMaskedPair>> many2many_passthrough_impl(
                    op_T &&op,
                    const at::ArrayRef<self_T> &tensors,
                    Args &&... args) {
                auto output_data = op.call(utils::get_data(tensors), args...);
                auto input_masks = utils::get_mask(tensors);
                bool has_any_mask = false;
                for (const auto i: c10::irange(input_masks.size())) {
                    if (input_masks[i].has_value()) {
                        has_any_mask = true;
                        break;
                    }
                }
                if (has_any_mask) {
                    std::vector<at::Tensor> input_masks_;
                    input_masks_.reserve(input_masks.size());
                    for (const auto i: c10::irange(input_masks.size())) {
                        input_masks_.emplace_back(input_masks[i].value_or(
                                at::ones_like(utils::get_data(tensors[i]),
                                              at::TensorOptions().dtype(at::kBool))));
                    }
                    auto output_masks = op.call(input_masks_, args...);
                    return utils::wrap_masked_pair(output_data, output_masks);
                }
                return utils::wrap_masked_pair(output_data, input_masks);
            }
        }

        // to
        namespace impl {
            template<typename self_T>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> to_impl(
                    const self_T &self,
                    c10::optional<at::ScalarType> dtype,
                    c10::optional<at::Layout> layout,
                    c10::optional<at::Device> device,
                    c10::optional<bool> pin_memory,
                    bool non_blocking,
                    bool copy,
                    c10::optional<at::MemoryFormat> memory_format) {
                auto self_data = utils::get_data(self);
                bool needs_copy = copy ||
                                  (dtype.has_value() && dtype.value() != self_data.scalar_type()) ||
                                  (layout.has_value() && layout.value() != self_data.layout()) ||
                                  (device.has_value() && device.value() != self_data.device());
                if (!needs_copy)
                    return masked_pair(self);
                auto self_mask = utils::get_mask(self);
                auto output_data = at::_ops::to_dtype_layout::call(
                        self_data, dtype, layout, device, pin_memory, non_blocking, copy, memory_format);
                auto output_mask = self_mask.has_value() ? at::_ops::to_dtype_layout::call(
                        self_mask.value(), {}, layout, device, pin_memory, non_blocking, copy, memory_format)
                                                         : c10::optional<at::Tensor>{};
                return masked_pair(output_data, output_mask);
            }

            template<typename self_T>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> to_impl(
                    const self_T &self,
                    at::Device device,
                    at::ScalarType dtype,
                    bool non_blocking,
                    bool copy,
                    c10::optional<at::MemoryFormat> memory_format) {
                auto self_data = utils::get_data(self);
                bool needs_copy = copy ||
                                  (dtype != self_data.scalar_type()) ||
                                  (device != self_data.device());
                if (!needs_copy)
                    return masked_pair(self);
                auto self_mask = utils::get_mask(self);
                auto output_data = at::_ops::to_device::call(
                        self_data, device, dtype, non_blocking, copy, memory_format);
                auto output_mask = self_mask.has_value() ? at::_ops::to_device::call(
                        self_mask.value(), device, {}, non_blocking, copy, memory_format)
                                                         : c10::optional<at::Tensor>{};
                return masked_pair(output_data, output_mask);
            }

            template<typename self_T>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> to_impl(
                    const self_T &self,
                    at::ScalarType dtype,
                    bool non_blocking,
                    bool copy,
                    c10::optional<at::MemoryFormat> memory_format) {
                auto self_data = utils::get_data(self);
                bool needs_copy = copy || (dtype != self_data.scalar_type());
                if (!needs_copy)
                    return masked_pair(self);
                auto output_data = at::_ops::to_dtype::call(
                        self_data, dtype, non_blocking, copy, memory_format);
                auto output_mask = utils::get_mask<true>(self);
                return masked_pair(output_data, output_mask);
            }

            template<typename self_T, typename other_T>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> to_impl(
                    const self_T &self,
                    const other_T &other,
                    bool non_blocking,
                    bool copy,
                    c10::optional<at::MemoryFormat> memory_format) {
                auto self_data = utils::get_data(self);
                auto other_data = utils::get_data(other);
                bool needs_copy = copy ||
                                  (self_data.scalar_type() != other_data.scalar_type()) ||
                                  (self_data.layout() != other_data.layout()) ||
                                  (self_data.device() != other_data.device());
                if (!needs_copy)
                    return masked_pair(self);
                auto self_mask = utils::get_mask(self);
                auto output_data = at::_ops::to_other::call(
                        self_data, other_data, non_blocking, copy, memory_format);
                auto output_mask = self_mask.has_value() ? at::_ops::to_dtype_layout::call(
                        self_mask.value(), {}, output_data.layout(), output_data.device(),
                        {}, non_blocking, copy, memory_format)
                                                         : c10::optional<at::Tensor>{};
                return masked_pair(self_data, self_mask);
            }
        }

        c10::intrusive_ptr<TensorMaskedPair> to(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                c10::optional<at::ScalarType> dtype,
                c10::optional<at::Layout> layout,
                c10::optional<at::Device> device,
                c10::optional<bool> pin_memory,
                bool non_blocking,
                bool copy,
                c10::optional<at::MemoryFormat> memory_format) {
            return impl::to_impl(self, dtype, layout, device, pin_memory, non_blocking, copy, memory_format);
        }

        c10::intrusive_ptr<TensorMaskedPair> to(
                const at::Tensor &self,
                c10::optional<at::ScalarType> dtype,
                c10::optional<at::Layout> layout,
                c10::optional<at::Device> device,
                c10::optional<bool> pin_memory,
                bool non_blocking,
                bool copy,
                c10::optional<at::MemoryFormat> memory_format) {
            return impl::to_impl(self, dtype, layout, device, pin_memory, non_blocking, copy, memory_format);
        }

        c10::intrusive_ptr<TensorMaskedPair> to(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                at::Device device,
                at::ScalarType dtype,
                bool non_blocking,
                bool copy,
                c10::optional<at::MemoryFormat> memory_format) {
            return impl::to_impl(self, device, dtype, non_blocking, copy, memory_format);
        }

        c10::intrusive_ptr<TensorMaskedPair> to(
                const at::Tensor &self,
                at::Device device,
                at::ScalarType dtype,
                bool non_blocking,
                bool copy,
                c10::optional<at::MemoryFormat> memory_format) {
            return impl::to_impl(self, device, dtype, non_blocking, copy, memory_format);
        }

        c10::intrusive_ptr<TensorMaskedPair> to(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                at::ScalarType dtype,
                bool non_blocking,
                bool copy,
                c10::optional<at::MemoryFormat> memory_format) {
            return impl::to_impl(self, dtype, non_blocking, copy, memory_format);
        }

        c10::intrusive_ptr<TensorMaskedPair> to(
                const at::Tensor &self,
                at::ScalarType dtype,
                bool non_blocking,
                bool copy,
                c10::optional<at::MemoryFormat> memory_format) {
            return impl::to_impl(self, dtype, non_blocking, copy, memory_format);
        }

        c10::intrusive_ptr<TensorMaskedPair> to(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                const_intrusive_ptr_arg_t<TensorMaskedPair> other,
                bool non_blocking,
                bool copy,
                c10::optional<at::MemoryFormat> memory_format) {
            return impl::to_impl(self, other, non_blocking, copy, memory_format);
        }

        c10::intrusive_ptr<TensorMaskedPair> to(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                const at::Tensor &other,
                bool non_blocking,
                bool copy,
                c10::optional<at::MemoryFormat> memory_format) {
            return impl::to_impl(self, other, non_blocking, copy, memory_format);
        }

        c10::intrusive_ptr<TensorMaskedPair> to(
                const at::Tensor &self,
                const_intrusive_ptr_arg_t<TensorMaskedPair> other,
                bool non_blocking,
                bool copy,
                c10::optional<at::MemoryFormat> memory_format) {
            return impl::to_impl(self, other, non_blocking, copy, memory_format);
        }

        c10::intrusive_ptr<TensorMaskedPair> to(
                const at::Tensor &self,
                const at::Tensor &other,
                bool non_blocking,
                bool copy,
                c10::optional<at::MemoryFormat> memory_format) {
            return impl::to_impl(self, other, non_blocking, copy, memory_format);
        }

        c10::intrusive_ptr<TensorMaskedPair> cpu(const_intrusive_ptr_arg_t<TensorMaskedPair> self) {
            return impl::to_impl(self, {}, {}, at::kCPU, {}, false, false, {});
        }

        c10::intrusive_ptr<TensorMaskedPair> cpu(const at::Tensor &self) {
            return impl::to_impl(self, {}, {}, at::kCPU, {}, false, false, {});
        }

        c10::intrusive_ptr<TensorMaskedPair> cuda(const_intrusive_ptr_arg_t<TensorMaskedPair> self) {
            return impl::to_impl(self, {}, {}, at::kCUDA, {}, false, false, {});
        }

        c10::intrusive_ptr<TensorMaskedPair> cuda(const at::Tensor &self) {
            return impl::to_impl(self, {}, {}, at::kCUDA, {}, false, false, {});
        }

        // one to one
        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                clone, at::_ops::clone(),
                c10::optional<at::MemoryFormat>, memory_format)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                contiguous, at::_ops::contiguous(),
                at::MemoryFormat, memory_format)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(
                detach, at::_ops::detach(), at::_ops::detach_())

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                detach_copy, at::_ops::detach_copy())

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                atleast_1d, at::_ops::atleast_1d())

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                atleast_2d, at::_ops::atleast_2d())

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                atleast_3d, at::_ops::atleast_3d())

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                diag, at::_ops::diag(),
                int64_t, diagonal)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                diag_embed, at::_ops::diag_embed(),
                int64_t, offset, int64_t, dim1, int64_t, dim2)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                diagflat, at::_ops::diagflat(),
                int64_t, offset)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                diagonal, at::_ops::diagonal(),
                int64_t, offset, int64_t, dim1, int64_t, dim2)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                linalg_diagonal, at::_ops::linalg_diagonal(),
                int64_t, offset, int64_t, dim1, int64_t, dim2)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                narrow, at::_ops::narrow(),
                int64_t, dim, c10::SymInt, start, c10::SymInt, length)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                narrow, at::_ops::narrow_Tensor(),
                int64_t, dim, const at::Tensor &, start, c10::SymInt, length)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                narrow_copy, at::_ops::narrow_copy(),
                int64_t, dim, c10::SymInt, start, c10::SymInt, length)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                select, at::_ops::select_int(),
                int64_t, dim, c10::SymInt, index)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                repeat, at::_ops::repeat(),
                c10::SymIntArrayRef, repeats)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                repeat_interleave, at::_ops::repeat_interleave_self_int(),
                c10::SymInt, repeats,
                c10::optional<int64_t>, dim,
                c10::optional<int64_t>, output_size)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                repeat_interleave, at::_ops::repeat_interleave_self_Tensor(),
                const at::Tensor &, repeats,
                c10::optional<int64_t>, dim,
                c10::optional<int64_t>, output_size)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                tile, at::_ops::tile(),
                c10::SymIntArrayRef, dims)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                ravel, at::_ops::ravel())

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                flatten, at::_ops::flatten_using_ints(),
                int64_t, start_dim, int64_t, end_dim)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                unflatten, at::_ops::unflatten_int(),
                int64_t, dim, c10::SymIntArrayRef, sizes)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                broadcast_to, at::_ops::broadcast_to(),
                c10::SymIntArrayRef, size)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                expand, at::_ops::expand(),
                c10::SymIntArrayRef, size,
                bool, implicit)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                expand_as, at::_ops::expand_as(),
                const at::Tensor &, other)

        c10::intrusive_ptr<TensorMaskedPair> expand_as(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                const_intrusive_ptr_arg_t<TensorMaskedPair> other) {
            static const auto op = at::_ops::expand_as();
            return impl::one2one_passthrough_impl(op, self, utils::get_data(other));
        }

        c10::intrusive_ptr<TensorMaskedPair> expand_as(
                const at::Tensor &self,
                const_intrusive_ptr_arg_t<TensorMaskedPair> other) {
            static const auto op = at::_ops::expand_as();
            return impl::one2one_passthrough_impl(op, self, utils::get_data(other));
        }

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                reshape, at::_ops::reshape(),
                c10::SymIntArrayRef, size)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                reshape_as, at::_ops::reshape_as(),
                const at::Tensor &, other)

        c10::intrusive_ptr<TensorMaskedPair> reshape_as(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                const_intrusive_ptr_arg_t<TensorMaskedPair> other) {
            static const auto op = at::_ops::reshape_as();
            return impl::one2one_passthrough_impl(op, self, utils::get_data(other));
        }

        c10::intrusive_ptr<TensorMaskedPair> reshape_as(
                const at::Tensor &self,
                const_intrusive_ptr_arg_t<TensorMaskedPair> other) {
            static const auto op = at::_ops::reshape_as();
            return impl::one2one_passthrough_impl(op, self, utils::get_data(other));
        }

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                view, at::_ops::view(),
                c10::SymIntArrayRef, size)

        c10::intrusive_ptr<TensorMaskedPair> view(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                at::ScalarType dtype) {
            static const auto op = at::_ops::view_dtype();
            return impl::one2one_passthrough_impl<false>(op, self, dtype);
        }

        c10::intrusive_ptr<TensorMaskedPair> view(
                const at::Tensor &self,
                at::ScalarType dtype) {
            static const auto op = at::_ops::view_dtype();
            return impl::one2one_passthrough_impl<false>(op, self, dtype);
        }

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                view_as, at::_ops::view_as(),
                const at::Tensor &, other)

        c10::intrusive_ptr<TensorMaskedPair> view_as(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                const_intrusive_ptr_arg_t<TensorMaskedPair> other) {
            static const auto op = at::_ops::view_as();
            return impl::one2one_passthrough_impl(op, self, utils::get_data(other));
        }

        c10::intrusive_ptr<TensorMaskedPair> view_as(
                const at::Tensor &self,
                const_intrusive_ptr_arg_t<TensorMaskedPair> other) {
            static const auto op = at::_ops::view_as();
            return impl::one2one_passthrough_impl(op, self, utils::get_data(other));
        }

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(
                squeeze, at::_ops::squeeze(), at::_ops::squeeze_())

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                squeeze, at::_ops::squeeze_dim(), at::_ops::squeeze__dim(),
                int64_t, dim)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                squeeze, at::_ops::squeeze_dims(), at::_ops::squeeze__dims(),
                at::IntArrayRef, dim)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                unsqueeze, at::_ops::unsqueeze(), at::_ops::unsqueeze_(),
                int64_t, dim)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                matrix_H, at::_ops::matrix_H())

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                moveaxis, at::_ops::moveaxis_intlist(),
                at::IntArrayRef, source, at::IntArrayRef, destination)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                moveaxes, at::_ops::moveaxis_intlist(),
                at::IntArrayRef, source, at::IntArrayRef, destination)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                moveaxis, at::_ops::moveaxis_int(),
                int64_t, source, int64_t, destination)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                moveaxes, at::_ops::moveaxis_int(),
                int64_t, source, int64_t, destination)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                movedim, at::_ops::movedim_intlist(),
                at::IntArrayRef, source, at::IntArrayRef, destination)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                movedims, at::_ops::movedim_intlist(),
                at::IntArrayRef, source, at::IntArrayRef, destination)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                movedim, at::_ops::movedim_int(),
                int64_t, source, int64_t, destination)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                movedims, at::_ops::movedim_int(),
                int64_t, source, int64_t, destination)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                swapaxis, at::_ops::swapaxes(), at::_ops::swapaxes_(),
                int64_t, axis0, int64_t, axis1)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                swapaxes, at::_ops::swapaxes(), at::_ops::swapaxes_(),
                int64_t, axis0, int64_t, axis1)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                swapdim, at::_ops::swapdims(), at::_ops::swapdims_(),
                int64_t, axis0, int64_t, axis1)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                swapdims, at::_ops::swapdims(), at::_ops::swapdims_(),
                int64_t, axis0, int64_t, axis1)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(
                t, at::_ops::t(), at::_ops::t_())

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                transpose, at::_ops::transpose_int(), at::_ops::transpose_(),
                int64_t, dim0, int64_t, dim1)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                permute, at::_ops::permute(),
                at::IntArrayRef, dims)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                permute_copy, at::_ops::permute_copy(),
                at::IntArrayRef, dims)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                take, at::_ops::take(),
                const at::Tensor &, index)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                take_along_dim, at::_ops::take_along_dim(),
                const at::Tensor &, indices, c10::optional<int64_t>, dim)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                gather, at::_ops::gather(),
                int64_t, dim, const at::Tensor &, index, bool, sparse_grad)

        PT_DEFINE_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                unfold, at::_ops::unfold(),
                int64_t, dimension, int64_t, size, int64_t, step)

        // one to many
        PT_DEFINE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                chunk, at::_ops::chunk(),
                int64_t, chunks, int64_t, dim)

        PT_DEFINE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                split, at::_ops::split_Tensor(),
                c10::SymInt, split_size, int64_t, dim)

        PT_DEFINE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                split, at::_ops::split_sizes(),
                c10::SymIntArrayRef, split_size, int64_t, dim)

        PT_DEFINE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                split_with_sizes, at::_ops::split_with_sizes(),
                c10::SymIntArrayRef, split_sizes, int64_t, dim)

        PT_DEFINE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                split_copy, at::_ops::split_copy_Tensor(),
                c10::SymInt, split_size, int64_t, dim)

        PT_DEFINE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                split_with_sizes_copy, at::_ops::split_with_sizes_copy(),
                c10::SymIntArrayRef, split_sizes, int64_t, dim)

        PT_DEFINE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                dsplit, at::_ops::dsplit_int(),
                int64_t, sections)

        PT_DEFINE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                dsplit, at::_ops::dsplit_array(),
                at::IntArrayRef, indices)

        PT_DEFINE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                hsplit, at::_ops::hsplit_int(),
                int64_t, sections)

        PT_DEFINE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                hsplit, at::_ops::hsplit_array(),
                at::IntArrayRef, indices)

        PT_DEFINE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                vsplit, at::_ops::vsplit_int(),
                int64_t, sections)

        PT_DEFINE_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                vsplit, at::_ops::vsplit_array(),
                at::IntArrayRef, indices)

        // many to one
        PT_DEFINE_MANY2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                cat, at::_ops::cat(),
                int64_t, dim)

        PT_DEFINE_MANY2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                row_stack, at::_ops::row_stack())

        PT_DEFINE_MANY2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                column_stack, at::_ops::column_stack())

        PT_DEFINE_MANY2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                hstack, at::_ops::hstack())

        PT_DEFINE_MANY2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                vstack, at::_ops::vstack())

        // many to many
        PT_DEFINE_MANY2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                atleast_1d, at::_ops::atleast_1d_Sequence())

        PT_DEFINE_MANY2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                atleast_2d, at::_ops::atleast_2d_Sequence())

        PT_DEFINE_MANY2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                atleast_3d, at::_ops::atleast_3d_Sequence())

        PT_DEFINE_MANY2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                broadcast_tensors, at::_ops::broadcast_tensors())

        PT_DEFINE_MANY2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                meshgrid, at::_ops::meshgrid())

        PT_DEFINE_MANY2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                meshgrid, at::_ops::meshgrid_indexing(),
                c10::string_view, indexing)

        TORCH_LIBRARY_FRAGMENT(partialtorch, m) {
            // to
            m.def(utils::FunctionSchemaBuilder("to").overload("MaskedPair_dtype_layout")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .vararg()
                          .arg<c10::optional<at::ScalarType>>("dtype", "None")
                          .arg<c10::optional<at::Layout>>("layout", "None")
                          .arg<c10::optional<at::Device>>("device", "None")
                          .arg<c10::optional<bool>>("pin_memory", "None")
                          .arg<bool>("non_blocking", "False")
                          .arg<bool>("copy", "False")
                          .arg<c10::optional<at::MemoryFormat>>("memory_format", "None")
                          .ret<c10::intrusive_ptr<TensorMaskedPair>>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          c10::optional<at::ScalarType>,
                          c10::optional<at::Layout>,
                          c10::optional<at::Device>,
                          c10::optional<bool>,
                          bool, bool,
                          c10::optional<at::MemoryFormat>)>(to)));
            m.def(utils::FunctionSchemaBuilder("to").overload("Tensor_dtype_layout")
                          .arg<const at::Tensor &>("self")
                          .vararg()
                          .arg<c10::optional<at::ScalarType>>("dtype", "None")
                          .arg<c10::optional<at::Layout>>("layout", "None")
                          .arg<c10::optional<at::Device>>("device", "None")
                          .arg<c10::optional<bool>>("pin_memory", "None")
                          .arg<bool>("non_blocking", "False")
                          .arg<bool>("copy", "False")
                          .arg<c10::optional<at::MemoryFormat>>("memory_format", "None")
                          .ret<c10::intrusive_ptr<TensorMaskedPair>>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          c10::optional<at::ScalarType>,
                          c10::optional<at::Layout>,
                          c10::optional<at::Device>,
                          c10::optional<bool>,
                          bool, bool,
                          c10::optional<at::MemoryFormat>)>(to)));

            m.def(utils::FunctionSchemaBuilder("to").overload("MaskedPair_device")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .arg<at::Device>("device")
                          .arg<at::ScalarType>("dtype")
                          .arg<bool>("non_blocking", "False")
                          .arg<bool>("copy", "False")
                          .arg<c10::optional<at::MemoryFormat>>("memory_format", "None")
                          .ret<c10::intrusive_ptr<TensorMaskedPair>>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          at::Device,
                          at::ScalarType,
                          bool, bool,
                          c10::optional<at::MemoryFormat>)>(to)));
            m.def(utils::FunctionSchemaBuilder("to").overload("Tensor_device")
                          .arg<const at::Tensor &>("self")
                          .arg<at::Device>("device")
                          .arg<at::ScalarType>("dtype")
                          .arg<bool>("non_blocking", "False")
                          .arg<bool>("copy", "False")
                          .arg<c10::optional<at::MemoryFormat>>("memory_format", "None")
                          .ret<c10::intrusive_ptr<TensorMaskedPair>>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          at::Device,
                          at::ScalarType,
                          bool, bool,
                          c10::optional<at::MemoryFormat>)>(to)));

            m.def(utils::FunctionSchemaBuilder("to").overload("MaskedPair_dtype")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .arg<at::ScalarType>("dtype")
                          .arg<bool>("non_blocking", "False")
                          .arg<bool>("copy", "False")
                          .arg<c10::optional<at::MemoryFormat>>("memory_format", "None")
                          .ret<c10::intrusive_ptr<TensorMaskedPair>>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          at::ScalarType,
                          bool, bool,
                          c10::optional<at::MemoryFormat>)>(to)));
            m.def(utils::FunctionSchemaBuilder("to").overload("Tensor_dtype")
                          .arg<const at::Tensor &>("self")
                          .arg<at::ScalarType>("dtype")
                          .arg<bool>("non_blocking", "False")
                          .arg<bool>("copy", "False")
                          .arg<c10::optional<at::MemoryFormat>>("memory_format", "None")
                          .ret<c10::intrusive_ptr<TensorMaskedPair>>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          at::ScalarType,
                          bool, bool,
                          c10::optional<at::MemoryFormat>)>(to)));

            m.def(utils::FunctionSchemaBuilder("to").overload("MaskedPair_MaskedPair_other")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("other")
                          .arg<bool>("non_blocking", "False")
                          .arg<bool>("copy", "False")
                          .arg<c10::optional<at::MemoryFormat>>("memory_format", "None")
                          .ret<c10::intrusive_ptr<TensorMaskedPair>>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          bool, bool,
                          c10::optional<at::MemoryFormat>)>(to)));
            m.def(utils::FunctionSchemaBuilder("to").overload("MaskedPair_Tensor_other")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .arg<const at::Tensor &>("other")
                          .arg<bool>("non_blocking", "False")
                          .arg<bool>("copy", "False")
                          .arg<c10::optional<at::MemoryFormat>>("memory_format", "None")
                          .ret<c10::intrusive_ptr<TensorMaskedPair>>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          const at::Tensor &,
                          bool, bool,
                          c10::optional<at::MemoryFormat>)>(to)));
            m.def(utils::FunctionSchemaBuilder("to").overload("Tensor_MaskedPair_other")
                          .arg<const at::Tensor &>("self")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("other")
                          .arg<bool>("non_blocking", "False")
                          .arg<bool>("copy", "False")
                          .arg<c10::optional<at::MemoryFormat>>("memory_format", "None")
                          .ret<c10::intrusive_ptr<TensorMaskedPair>>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          bool, bool,
                          c10::optional<at::MemoryFormat>)>(to)));
            m.def(utils::FunctionSchemaBuilder("to").overload("Tensor_Tensor_other")
                          .arg<const at::Tensor &>("self")
                          .arg<const at::Tensor &>("other")
                          .arg<bool>("non_blocking", "False")
                          .arg<bool>("copy", "False")
                          .arg<c10::optional<at::MemoryFormat>>("memory_format", "None")
                          .ret<c10::intrusive_ptr<TensorMaskedPair>>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          const at::Tensor &,
                          bool, bool,
                          c10::optional<at::MemoryFormat>)>(to)));

            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                    cpu,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                    cuda,)

            // one to one
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    clone, ,
                    c10::optional<at::MemoryFormat>, memory_format, None)

            m.def(utils::FunctionSchemaBuilder("contiguous").overload("MaskedPair")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .vararg()
                          .arg<at::MemoryFormat>("memory_format", "contiguous_format")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          at::MemoryFormat)>(contiguous)));
            m.def(utils::FunctionSchemaBuilder("contiguous").overload("Tensor")
                          .arg<const at::Tensor &>("self")
                          .vararg()
                          .arg<at::MemoryFormat>("memory_format", "contiguous_format")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          at::MemoryFormat)>(contiguous)));

            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(
                    detach,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                    detach_copy,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                    atleast_1d,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                    atleast_2d,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                    atleast_3d,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    diag, ,
                    int64_t, diagonal, 0)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                    diag_embed, ,
                    int64_t, offset, 0,
                    int64_t, dim1, 0,
                    int64_t, dim2, 1)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    diagflat, ,
                    int64_t, offset, 0)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                    diagonal, ,
                    int64_t, offset, 0,
                    int64_t, dim1, 0,
                    int64_t, dim2, 1)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                    linalg_diagonal, ,
                    int64_t, offset, 0,
                    int64_t, dim1, -2,
                    int64_t, dim2, -1)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                    narrow, ,
                    int64_t, dim, ,
                    c10::SymInt, start, ,
                    c10::SymInt, length,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                    narrow, Tensor,
                    int64_t, dim, ,
                    const at::Tensor &, start, ,
                    c10::SymInt, length,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                    narrow_copy, ,
                    int64_t, dim, ,
                    c10::SymInt, start, ,
                    c10::SymInt, length,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    select, ,
                    int64_t, dim, ,
                    c10::SymInt, index,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    repeat, ,
                    c10::SymIntArrayRef, repeats,)

            m.def(utils::FunctionSchemaBuilder("repeat_interleave").add_overload("MaskedPair").add_overload("int")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .arg<c10::SymInt>("repeats")
                          .arg<c10::optional<int64_t>>("dim")
                          .vararg()
                          .arg<c10::optional<int64_t>>("output_size", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair>(*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          c10::SymInt,
                          c10::optional<int64_t>,
                          c10::optional<int64_t>)>(repeat_interleave)));
            m.def(utils::FunctionSchemaBuilder("repeat_interleave").add_overload("Tensor").add_overload("int")
                          .arg<const at::Tensor &>("self")
                          .arg<c10::SymInt>("repeats")
                          .arg<c10::optional<int64_t>>("dim")
                          .vararg()
                          .arg<c10::optional<int64_t>>("output_size", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair>(*)(
                          const at::Tensor &,
                          c10::SymInt,
                          c10::optional<int64_t>,
                          c10::optional<int64_t>)>(repeat_interleave)));

            m.def(utils::FunctionSchemaBuilder("repeat_interleave").add_overload("MaskedPair").add_overload("Tensor")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .arg<const at::Tensor &>("repeats")
                          .arg<c10::optional<int64_t>>("dim")
                          .vararg()
                          .arg<c10::optional<int64_t>>("output_size", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair>(*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          const at::Tensor &,
                          c10::optional<int64_t>,
                          c10::optional<int64_t>)>(repeat_interleave)));
            m.def(utils::FunctionSchemaBuilder("repeat_interleave").add_overload("Tensor").add_overload("Tensor")
                          .arg<const at::Tensor &>("self")
                          .arg<const at::Tensor &>("repeats")
                          .arg<c10::optional<int64_t>>("dim")
                          .vararg()
                          .arg<c10::optional<int64_t>>("output_size", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair>(*)(
                          const at::Tensor &,
                          const at::Tensor &,
                          c10::optional<int64_t>,
                          c10::optional<int64_t>)>(repeat_interleave)));

            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    tile, ,
                    c10::SymIntArrayRef, dims,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                    ravel,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    flatten, ,
                    int64_t, start_dim, 0,
                    int64_t, end_dim, 0)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    unflatten, ,
                    int64_t, dim, ,
                    c10::SymIntArrayRef, sizes,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    broadcast_to, ,
                    c10::SymIntArrayRef, size,)

            m.def(utils::FunctionSchemaBuilder("expand").add_overload("MaskedPair")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self")
                          .arg<c10::SymIntArrayRef>("size")
                          .vararg()
                          .arg<bool>("implicit", "False")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair>(*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>, c10::SymIntArrayRef, bool)>(expand)));
            m.def(utils::FunctionSchemaBuilder("expand").add_overload("Tensor")
                          .arg<const at::Tensor &>("self")
                          .arg<c10::SymIntArrayRef>("size")
                          .vararg()
                          .arg<bool>("implicit", "False")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair>(*)(
                          const at::Tensor &, c10::SymIntArrayRef, bool)>(expand)));

            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    expand_as, Tensor,
                    const at::Tensor &, other,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    expand_as, MaskedPair,
                    const_intrusive_ptr_arg_t<TensorMaskedPair>, other,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    reshape, ,
                    c10::SymIntArrayRef, size,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    reshape_as, Tensor,
                    const at::Tensor &, other,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    reshape_as, MaskedPair,
                    const_intrusive_ptr_arg_t<TensorMaskedPair>, other,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    view, ,
                    c10::SymIntArrayRef, size,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    view, dtype,
                    at::ScalarType, dtype,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    view_as, Tensor,
                    const at::Tensor &, other,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    view_as, MaskedPair,
                    const_intrusive_ptr_arg_t<TensorMaskedPair>, other,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(
                    squeeze,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                    squeeze, dim,
                    int64_t, dim,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                    squeeze, dims,
                    at::IntArrayRef, dim,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                    unsqueeze, ,
                    int64_t, dim,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                    matrix_H,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    moveaxis, intlist,
                    at::IntArrayRef, source, ,
                    at::IntArrayRef, destination,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    moveaxes, intlist,
                    at::IntArrayRef, source, ,
                    at::IntArrayRef, destination,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    moveaxis, int,
                    int64_t, source, ,
                    int64_t, destination,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    moveaxes, int,
                    int64_t, source, ,
                    int64_t, destination,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    movedim, intlist,
                    at::IntArrayRef, source, ,
                    at::IntArrayRef, destination,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    movedims, intlist,
                    at::IntArrayRef, source, ,
                    at::IntArrayRef, destination,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    movedim, int,
                    int64_t, source, ,
                    int64_t, destination,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    movedims, int,
                    int64_t, source, ,
                    int64_t, destination,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                    swapaxis, ,
                    int64_t, axis0, ,
                    int64_t, axis1,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                    swapaxes, ,
                    int64_t, axis0, ,
                    int64_t, axis1,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                    swapdim, ,
                    int64_t, axis0, ,
                    int64_t, axis1,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                    swapdims, ,
                    int64_t, axis0, ,
                    int64_t, axis1,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(
                    t,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                    transpose, ,
                    int64_t, dim0, ,
                    int64_t, dim1,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    permute, ,
                    at::IntArrayRef, dims,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    permute_copy, ,
                    at::IntArrayRef, dims,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    take, ,
                    const at::Tensor &, index,)
            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    take_along_dim, ,
                    const at::Tensor &, indices, , c10::optional<int64_t>, dim, None)

            m.def(utils::FunctionSchemaBuilder("gather").overload("MaskedPair")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("self").arg<int64_t>("dim")
                          .arg<const at::Tensor &>("index")
                          .vararg().arg<bool>("sparse_grad", "False")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair>(*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          int64_t,
                          const at::Tensor &,
                          bool)>(gather)));
            m.def(utils::FunctionSchemaBuilder("gather").overload("Tensor")
                          .arg<const at::Tensor &>("self")
                          .arg<int64_t>("dim")
                          .arg<const at::Tensor &>("index")
                          .vararg().arg<bool>("sparse_grad", "False")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair>(*)(
                          const at::Tensor &,
                          int64_t,
                          const at::Tensor &,
                          bool)>(gather)));

            PT_REGISTER_ONE2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                    unfold, ,
                    int64_t, dimension, , int64_t, size, , int64_t, step,)

            // one to many
            PT_REGISTER_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    chunk, ,
                    int64_t, chunks, ,
                    int64_t, dim, 0)
            PT_REGISTER_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    split, Tensor,
                    c10::SymInt, split_size, ,
                    int64_t, dim, 0)
            PT_REGISTER_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    split, sizes,
                    c10::SymIntArrayRef, split_size, ,
                    int64_t, dim, 0)
            PT_REGISTER_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    split_with_sizes, ,
                    c10::SymIntArrayRef, split_sizes, ,
                    int64_t, dim, 0)
            PT_REGISTER_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    split_copy, Tensor,
                    c10::SymInt, split_size, ,
                    int64_t, dim, 0)
            PT_REGISTER_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                    split_with_sizes_copy, ,
                    c10::SymIntArrayRef, split_sizes, ,
                    int64_t, dim, 0)
            PT_REGISTER_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    dsplit, int,
                    int64_t, sections,)
            PT_REGISTER_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    dsplit, array,
                    at::IntArrayRef, indices,)
            PT_REGISTER_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    hsplit, int,
                    int64_t, sections,)
            PT_REGISTER_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    hsplit, array,
                    at::IntArrayRef, indices,)
            PT_REGISTER_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    vsplit, int,
                    int64_t, sections,)
            PT_REGISTER_ONE2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    vsplit, array,
                    at::IntArrayRef, indices,)

            // many to one
            PT_REGISTER_MANY2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    cat, ,
                    int64_t, dim,)
            PT_REGISTER_MANY2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                    row_stack,)
            PT_REGISTER_MANY2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                    column_stack,)
            PT_REGISTER_MANY2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                    hstack,)
            PT_REGISTER_MANY2ONE_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                    vstack,)

            // many to many
            PT_REGISTER_MANY2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                    atleast_1d, Sequence)
            PT_REGISTER_MANY2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                    atleast_2d, Sequence)
            PT_REGISTER_MANY2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                    atleast_3d, Sequence)
            PT_REGISTER_MANY2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                    broadcast_tensors,)
            PT_REGISTER_MANY2MANY_PASSTHROUGH_OPS_FORALL_TENSOR_OVERLOADS(
                    meshgrid,)

            m.def(utils::FunctionSchemaBuilder("meshgrid").add_overload("MaskedPair").add_overload("indexing")
                          .arg<const at::ArrayRef<c10::intrusive_ptr<TensorMaskedPair>>>("self")
                          .vararg()
                          .arg<c10::string_view>("indexing", "")
                          .ret<std::vector<TensorMaskedPair>>().schema().c_str(),
                  TORCH_FN(static_cast<std::vector<c10::intrusive_ptr<TensorMaskedPair>> (*)(
                          const at::ArrayRef<c10::intrusive_ptr<TensorMaskedPair>>, c10::string_view)>(meshgrid)));
            m.def(utils::FunctionSchemaBuilder("meshgrid").add_overload("Tensor").add_overload("indexing")
                          .arg<const at::ArrayRef<at::Tensor>>("self")
                          .vararg()
                          .arg<c10::string_view>("indexing", "")
                          .ret<std::vector<TensorMaskedPair>>().schema().c_str(),
                  TORCH_FN(static_cast<std::vector<c10::intrusive_ptr<TensorMaskedPair>> (*)(
                          const at::ArrayRef<at::Tensor>, c10::string_view)>(meshgrid)));
        }
    }
}
