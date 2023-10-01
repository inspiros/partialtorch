#include <torch/library.h>

#include "../MaskedPair.h"
#include "utils/mask_utils.h"
#include "utils/schema_utils.h"
#include "utils/torch_library_utils.h"

// ~~~~~ ops definition macros ~~~~~
#define PT_DEFINE_TERNARY_OP(NAME, IMPL_OP, SELF_T, OTHER1_T, OTHER2_T)       \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                    \
    SELF_T self,                                                              \
    OTHER1_T other1,                                                          \
    OTHER2_T other2) {                                                        \
    static constexpr auto op = IMPL_OP;                                       \
    return impl::bitwise_ternary_impl(op, self, other1, other2);              \
}

#define PT_DEFINE_TERNARY_OP_WITH(NAME, IMPL_OP, SELF_T, OTHER1_T, OTHER2_T, ARG1_T, ARG1_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                      \
    SELF_T self,                                                                                \
    OTHER1_T other1,                                                                            \
    OTHER2_T other2,                                                                            \
    ARG1_T ARG1_NAME) {                                                                         \
    static constexpr auto op = IMPL_OP;                                                         \
    return impl::bitwise_ternary_impl(op, self, other1, other2, ARG1_NAME);                     \
}

#define PT_DEFINE_TERNARY_OP_WITH2(NAME, IMPL_OP, SELF_T, OTHER1_T, OTHER2_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                                          \
    SELF_T self,                                                                                                    \
    OTHER1_T other1,                                                                                                \
    OTHER2_T other2,                                                                                                \
    ARG1_T ARG1_NAME,                                                                                               \
    ARG2_T ARG2_NAME) {                                                                                             \
    static constexpr auto op = IMPL_OP;                                                                             \
    return impl::bitwise_ternary_impl(op, self, other1, other2, ARG1_NAME, ARG2_NAME);                              \
}

#define PT_DEFINE_TERNARY_OP_(NAME, IMPL_OP_, SELF_T, OTHER1_T, OTHER2_T)       \
c10::intrusive_ptr<TensorMaskedPair> NAME##_(                                   \
    SELF_T self,                                                                \
    OTHER1_T other1,                                                            \
    OTHER2_T other2) {                                                          \
    static constexpr auto op_ = IMPL_OP_;                                       \
    return impl::bitwise_ternary__impl(op_, self, other1, other2);              \
}

#define PT_DEFINE_TERNARY_OP__WITH(NAME, IMPL_OP_, SELF_T, OTHER1_T, OTHER2_T, ARG1_T, ARG1_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME##_(                                                     \
    SELF_T self,                                                                                  \
    OTHER1_T other1,                                                                              \
    OTHER2_T other2,                                                                              \
    ARG1_T ARG1_NAME) {                                                                           \
    static constexpr auto op_ = IMPL_OP_;                                                         \
    return impl::bitwise_ternary__impl(op_, self, other1, other2, ARG1_NAME);                     \
}

#define PT_DEFINE_TERNARY_OP__WITH2(NAME, IMPL_OP_, SELF_T, OTHER1_T, OTHER2_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME##_(                                                                         \
    SELF_T self,                                                                                                      \
    OTHER1_T other1,                                                                                                  \
    OTHER2_T other2,                                                                                                  \
    ARG1_T ARG1_NAME,                                                                                                 \
    ARG2_T ARG2_NAME) {                                                                                               \
    static constexpr auto op_ = IMPL_OP_;                                                                             \
    return impl::bitwise_ternary__impl(op_, self, other1, other2, ARG1_NAME, ARG2_NAME);                              \
}

#define PT_DEFINE_TERNARY_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP) \
PT_DEFINE_TERNARY_OP(NAME, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DEFINE_TERNARY_OP(NAME, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &)                          \
PT_DEFINE_TERNARY_OP(NAME, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>)                          \
PT_DEFINE_TERNARY_OP(NAME, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const at::Tensor &)                                                   \
PT_DEFINE_TERNARY_OP(NAME, IMPL_OP, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>)                          \
PT_DEFINE_TERNARY_OP(NAME, IMPL_OP, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &)                                                   \
PT_DEFINE_TERNARY_OP(NAME, IMPL_OP, const at::Tensor &, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>)                                                   \
PT_DEFINE_TERNARY_OP(NAME, IMPL_OP, const at::Tensor &, const at::Tensor &, const at::Tensor &)

#define PT_DEFINE_TERNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP, ARG1_T, ARG1_NAME) \
PT_DEFINE_TERNARY_OP_WITH(NAME, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME) \
PT_DEFINE_TERNARY_OP_WITH(NAME, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1_T, ARG1_NAME)                          \
PT_DEFINE_TERNARY_OP_WITH(NAME, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME)                          \
PT_DEFINE_TERNARY_OP_WITH(NAME, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const at::Tensor &, ARG1_T, ARG1_NAME)                                                   \
PT_DEFINE_TERNARY_OP_WITH(NAME, IMPL_OP, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME)                          \
PT_DEFINE_TERNARY_OP_WITH(NAME, IMPL_OP, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1_T, ARG1_NAME)                                                   \
PT_DEFINE_TERNARY_OP_WITH(NAME, IMPL_OP, const at::Tensor &, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME)                                                   \
PT_DEFINE_TERNARY_OP_WITH(NAME, IMPL_OP, const at::Tensor &, const at::Tensor &, const at::Tensor &, ARG1_T, ARG1_NAME)

#define PT_DEFINE_TERNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, IMPL_OP, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_TERNARY_OP_WITH2(NAME, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_TERNARY_OP_WITH2(NAME, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                          \
PT_DEFINE_TERNARY_OP_WITH2(NAME, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                          \
PT_DEFINE_TERNARY_OP_WITH2(NAME, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                                                   \
PT_DEFINE_TERNARY_OP_WITH2(NAME, IMPL_OP, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                          \
PT_DEFINE_TERNARY_OP_WITH2(NAME, IMPL_OP, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                                                   \
PT_DEFINE_TERNARY_OP_WITH2(NAME, IMPL_OP, const at::Tensor &, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                                                   \
PT_DEFINE_TERNARY_OP_WITH2(NAME, IMPL_OP, const at::Tensor &, const at::Tensor &, const at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)

#define PT_DEFINE_TERNARY_OPS__FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP_, IDENTITY_OP_) \
PT_DEFINE_TERNARY_OP_(NAME, IMPL_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DEFINE_TERNARY_OP_(NAME, IMPL_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &)                          \
PT_DEFINE_TERNARY_OP_(NAME, IMPL_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>)                          \
PT_DEFINE_TERNARY_OP_(NAME, IMPL_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const at::Tensor &)                                                   \
PT_DEFINE_TERNARY_OP_(NAME, IMPL_OP_, at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>)                          \
PT_DEFINE_TERNARY_OP_(NAME, IMPL_OP_, at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &)                                                   \
PT_DEFINE_TERNARY_OP_(NAME, IMPL_OP_, at::Tensor &, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>)                                                   \
PT_DEFINE_TERNARY_OP_(NAME, IMPL_OP_, at::Tensor &, const at::Tensor &, const at::Tensor &)

#define PT_DEFINE_TERNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP_, ARG1_T, ARG1_NAME) \
PT_DEFINE_TERNARY_OP__WITH(NAME, IMPL_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME) \
PT_DEFINE_TERNARY_OP__WITH(NAME, IMPL_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1_T, ARG1_NAME)                          \
PT_DEFINE_TERNARY_OP__WITH(NAME, IMPL_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME)                          \
PT_DEFINE_TERNARY_OP__WITH(NAME, IMPL_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const at::Tensor &, ARG1_T, ARG1_NAME)                                                   \
PT_DEFINE_TERNARY_OP__WITH(NAME, IMPL_OP_, at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME)                          \
PT_DEFINE_TERNARY_OP__WITH(NAME, IMPL_OP_, at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1_T, ARG1_NAME)                                                   \
PT_DEFINE_TERNARY_OP__WITH(NAME, IMPL_OP_, at::Tensor &, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME)                                                   \
PT_DEFINE_TERNARY_OP__WITH(NAME, IMPL_OP_, at::Tensor &, const at::Tensor &, const at::Tensor &, ARG1_T, ARG1_NAME)

#define PT_DEFINE_TERNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, IMPL_OP_, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_TERNARY_OP__WITH2(NAME, IMPL_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_TERNARY_OP__WITH2(NAME, IMPL_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                          \
PT_DEFINE_TERNARY_OP__WITH2(NAME, IMPL_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                          \
PT_DEFINE_TERNARY_OP__WITH2(NAME, IMPL_OP_, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, const at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                                                   \
PT_DEFINE_TERNARY_OP__WITH2(NAME, IMPL_OP_, at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                          \
PT_DEFINE_TERNARY_OP__WITH2(NAME, IMPL_OP_, at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                                                   \
PT_DEFINE_TERNARY_OP__WITH2(NAME, IMPL_OP_, at::Tensor &, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                                                   \
PT_DEFINE_TERNARY_OP__WITH2(NAME, IMPL_OP_, at::Tensor &, const at::Tensor &, const at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)

#define PT_DEFINE_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, IMPL_OP_, IDENTITY_OP_)  \
PT_DEFINE_TERNARY_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, IDENTITY_OP)                              \
PT_DEFINE_TERNARY_OPS__FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP_, IDENTITY_OP_)

#define PT_DEFINE_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP, IMPL_OP_, ARG1_T, ARG1_NAME)  \
PT_DEFINE_TERNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP, ARG1_T, ARG1_NAME)                             \
PT_DEFINE_TERNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, IMPL_OP_, ARG1_T, ARG1_NAME)

#define PT_DEFINE_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, IMPL_OP, IMPL_OP_, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)  \
PT_DEFINE_TERNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, IMPL_OP, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                             \
PT_DEFINE_TERNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, IMPL_OP_, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)

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
            template<typename op_T, typename self_T, typename other1_T, typename other2_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> bitwise_ternary_impl(
                    op_T &&op,
                    const self_T &self,
                    const other1_T &other1,
                    const other2_T &other2,
                    Args &&... args) {
                auto output_data = op.call(utils::get_data(self),
                                           utils::get_data(other1),
                                           utils::get_data(other2), args...);
                auto output_mask = utils::all_masks(utils::get_mask(self),
                                                    utils::get_mask(other1),
                                                    utils::get_mask(other2));
                return masked_pair(output_data, output_mask);
            }

            template<typename op__T, typename self_T, typename other1_T, typename other2_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> bitwise_ternary__impl(
                    op__T &&op_,
                    self_T &self,
                    const other1_T &other1,
                    const other2_T &other2,
                    Args &&... args) {
                auto self_data = utils::get_data(self);
                auto output_data = op_.call(self_data,
                                            utils::get_data(other1),
                                            utils::get_data(other2), args...);
                auto output_mask = utils::all_masks_(utils::get_mask(self),
                                                     utils::get_mask(other1),
                                                     utils::get_mask(other2));
                if constexpr (std::is_same_v<c10::base_t<self_T>, TensorMaskedPair>) {
                    return self;
                } else {
                    return masked_pair(output_data, output_mask);
                }
            }
        }

        PT_DEFINE_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                addcmul, at::_ops::addcmul(), at::_ops::addcmul_(),
                const at::Scalar &, value)

        PT_DEFINE_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                addcdiv, at::_ops::addcdiv(), at::_ops::addcdiv_(),
                const at::Scalar &, value)

        TORCH_LIBRARY_FRAGMENT(partialtorch, m) {
            PT_REGISTER_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                    addcmul, , tensor1, tensor2, ,
                    const at::Scalar &, alpha, 1)
            PT_REGISTER_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                    addcdiv, , tensor1, tensor2, ,
                    const at::Scalar &, alpha, 1)
        }
    }
}
