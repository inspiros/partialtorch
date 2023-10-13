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
#define PT_DEFINE_BINARY_OP(NAME, IMPL_OP, IDENTITY_OP, SELF_T, OTHER_T)        \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                      \
    SELF_T self,                                                                \
    OTHER_T other) {                                                            \
    static constexpr auto op = IMPL_OP;                                         \
    static constexpr auto fill_identity_op = IDENTITY_OP;                       \
    return impl::partial_binary_impl<false>(op, fill_identity_op, nullptr, self, other); \
}

#define PT_DEFINE_BINARY_OP_WITH(NAME, IMPL_OP, IDENTITY_OP, SELF_T, OTHER_T, ARG1_T, ARG1_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                       \
    SELF_T self,                                                                                 \
    OTHER_T other,                                                                               \
    ARG1_T ARG1_NAME) {                                                                          \
    static constexpr auto op = IMPL_OP;                                                          \
    static constexpr auto fill_identity_op = IDENTITY_OP;                                        \
    return impl::partial_binary_impl<false>(op, fill_identity_op, nullptr, self, other, ARG1_NAME); \
}

#define PT_DEFINE_BINARY_OP_WITH2(NAME, IMPL_OP, IDENTITY_OP, SELF_T, OTHER_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                                           \
    SELF_T self,                                                                                                     \
    OTHER_T other,                                                                                                   \
    ARG1_T ARG1_NAME,                                                                                                \
    ARG2_T ARG2_NAME) {                                                                                              \
    static constexpr auto op = IMPL_OP;                                                                              \
    static constexpr auto fill_identity_op = IDENTITY_OP;                                                            \
    return impl::partial_binary_impl<false>(op, fill_identity_op, nullptr, self, other, ARG1_NAME, ARG2_NAME);       \
}

#define PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(NAME, TENSOR_OP, IDENTITY_OP) \
PT_DEFINE_BINARY_OP(NAME, TENSOR_OP, IDENTITY_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DEFINE_BINARY_OP(NAME, TENSOR_OP, IDENTITY_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &)                          \
PT_DEFINE_BINARY_OP(NAME, TENSOR_OP, IDENTITY_OP, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>)                          \
PT_DEFINE_BINARY_OP(NAME, TENSOR_OP, IDENTITY_OP, const at::Tensor &, const at::Tensor &)

#define PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, TENSOR_OP, IDENTITY_OP, ARG1_T, ARG1_NAME) \
PT_DEFINE_BINARY_OP_WITH(NAME, TENSOR_OP, IDENTITY_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME) \
PT_DEFINE_BINARY_OP_WITH(NAME, TENSOR_OP, IDENTITY_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1_T, ARG1_NAME)                          \
PT_DEFINE_BINARY_OP_WITH(NAME, TENSOR_OP, IDENTITY_OP, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME)                          \
PT_DEFINE_BINARY_OP_WITH(NAME, TENSOR_OP, IDENTITY_OP, const at::Tensor &, const at::Tensor &, ARG1_T, ARG1_NAME)

#define PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, TENSOR_OP, IDENTITY_OP, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_BINARY_OP_WITH2(NAME, TENSOR_OP, IDENTITY_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_BINARY_OP_WITH2(NAME, TENSOR_OP, IDENTITY_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                          \
PT_DEFINE_BINARY_OP_WITH2(NAME, TENSOR_OP, IDENTITY_OP, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)                          \
PT_DEFINE_BINARY_OP_WITH2(NAME, TENSOR_OP, IDENTITY_OP, const at::Tensor &, const at::Tensor &, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)

// scaled ops
#define PT_DEFINE_SCALED_BINARY_OP(NAME, IMPL_OP, IDENTITY_OP, CANONICAL_SCALE, SELF_T, OTHER_T) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                       \
    SELF_T self,                                                                                 \
    OTHER_T other,                                                                               \
    bool scaled) {                                                                               \
    static constexpr auto op = IMPL_OP;                                                          \
    static constexpr auto fill_identity_op = IDENTITY_OP;                                        \
    static constexpr auto canonical_scale = CANONICAL_SCALE;                                     \
    if (!scaled)                                                                                 \
        return impl::partial_binary_impl<false>(op, fill_identity_op, nullptr, self, other);     \
    return impl::partial_binary_impl<true>(op, fill_identity_op, canonical_scale, self, other);  \
}

#define PT_DEFINE_SCALED_BINARY_OP_WITH(NAME, IMPL_OP, IDENTITY_OP, CANONICAL_SCALE, SELF_T, OTHER_T, ARG1_T, ARG1_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                                               \
    SELF_T self,                                                                                                         \
    OTHER_T other,                                                                                                       \
    ARG1_T ARG1_NAME,                                                                                                    \
    bool scaled) {                                                                                                       \
    static constexpr auto op = IMPL_OP;                                                                                  \
    static constexpr auto fill_identity_op = IDENTITY_OP;                                                                \
    static constexpr auto canonical_scale = CANONICAL_SCALE;                                                             \
    if (!scaled)                                                                                                         \
        return impl::partial_binary_impl<false>(op, fill_identity_op, nullptr, self, other, ARG1_NAME);                  \
    return impl::partial_binary_impl<true>(op, fill_identity_op, canonical_scale, self, other, ARG1_NAME);               \
}

#define PT_DEFINE_SCALED_BINARY_OPS_FORALL_TENSOR_OVERLOADS(NAME, TENSOR_OP, IDENTITY_OP, CANONICAL_SCALE) \
PT_DEFINE_SCALED_BINARY_OP(NAME, TENSOR_OP, IDENTITY_OP, CANONICAL_SCALE, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DEFINE_SCALED_BINARY_OP(NAME, TENSOR_OP, IDENTITY_OP, CANONICAL_SCALE, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &)                          \
PT_DEFINE_SCALED_BINARY_OP(NAME, TENSOR_OP, IDENTITY_OP, CANONICAL_SCALE, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>)                          \
PT_DEFINE_SCALED_BINARY_OP(NAME, TENSOR_OP, IDENTITY_OP, CANONICAL_SCALE, const at::Tensor &, const at::Tensor &)

#define PT_DEFINE_SCALED_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, TENSOR_OP, IDENTITY_OP, CANONICAL_SCALE, ARG1_T, ARG1_NAME) \
PT_DEFINE_SCALED_BINARY_OP_WITH(NAME, TENSOR_OP, IDENTITY_OP, CANONICAL_SCALE, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME) \
PT_DEFINE_SCALED_BINARY_OP_WITH(NAME, TENSOR_OP, IDENTITY_OP, CANONICAL_SCALE, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1_T, ARG1_NAME)                          \
PT_DEFINE_SCALED_BINARY_OP_WITH(NAME, TENSOR_OP, IDENTITY_OP, CANONICAL_SCALE, const at::Tensor &, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1_T, ARG1_NAME)                          \
PT_DEFINE_SCALED_BINARY_OP_WITH(NAME, TENSOR_OP, IDENTITY_OP, CANONICAL_SCALE, const at::Tensor &, const at::Tensor &, ARG1_T, ARG1_NAME)

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

namespace partialtorch {
    namespace ops {
        namespace impl {
            template<bool scaled = false, typename op_T, typename fill_identity_op_T, typename canonical_scale_T,
                    typename self_T, typename other_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> partial_binary_impl(
                    op_T &&op,
                    fill_identity_op_T &&fill_identity_op,
                    canonical_scale_T &&canonical_scale,
                    const self_T &self,
                    const other_T &other,
                    Args &&... args) {
                if (!utils::has_tensor_mask(self) && !utils::has_tensor_mask(other)) {
                    return masked_pair(op.call(utils::get_data(self), utils::get_data(other), args...));
                }
                auto output_data = op.call(fill_identity_op.call(self),
                                           fill_identity_op.call(other), args...);
                auto mask_ratio_options = output_data.options();
                at::Tensor mask_ratio;
                {
                    at::NoGradGuard g;
                    mask_ratio = op.call(
                            utils::get_tensor_mask(self, mask_ratio_options),
                            utils::get_tensor_mask(other, mask_ratio_options), args...);
                    if constexpr (scaled) {
                        if constexpr (std::is_same_v<std::base_t<canonical_scale_T>, nullptr_t>) {
                            utils::izero_ldiv_(
                                    mask_ratio,
                                    op.call(at::ones_like(utils::get_data(self)),
                                            at::ones_like(utils::get_data(other)), args...)
                            );
                        } else if constexpr (std::is_same_v<std::base_t<canonical_scale_T>, at::Scalar> ||
                                             std::is_same_v<std::base_t<canonical_scale_T>, at::Tensor> ||
                                             std::is_arithmetic_v<std::base_t<canonical_scale_T>>) {
                            utils::izero_ldiv_(mask_ratio, canonical_scale);
                        } else {
                            utils::izero_ldiv_(
                                    mask_ratio,
                                    canonical_scale.call(utils::get_data(self),
                                                         utils::get_data(other), args...)
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
        }

        // arithmetics
        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                partial_ger, at::_ops::ger(),
                utils::_ops::fill_identity_zeros())

        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                partial_mm, at::_ops::mm(),
                utils::_ops::fill_identity_zeros())

        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                partial_bmm, at::_ops::bmm(),
                utils::_ops::fill_identity_zeros())

        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                partial_matmul, at::_ops::matmul(),
                utils::_ops::fill_identity_zeros())

        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                linalg_partial_matmul, at::_ops::linalg_matmul(),
                utils::_ops::fill_identity_zeros())

        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                partial_mv, at::_ops::mv(),
                utils::_ops::fill_identity_zeros())

        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                partial_inner, at::_ops::inner(),
                utils::_ops::fill_identity_zeros())

        PT_DEFINE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                partial_outer, at::_ops::outer(),
                utils::_ops::fill_identity_zeros())

        // scaled arithmetics
        namespace impl {
            namespace _ops {
                struct mm_scale {
                    static inline int64_t call(const at::Tensor &self, const at::Tensor &) {
                        return self.size(-1);
                    }
                };
            }
        }

        // TODO: dot, vdot, cross

        PT_DEFINE_SCALED_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                partial_ger, at::_ops::ger(),
                utils::_ops::fill_identity_zeros(), 1)

        PT_DEFINE_SCALED_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                partial_mm, at::_ops::mm(),
                utils::_ops::fill_identity_zeros(), impl::_ops::mm_scale())

        PT_DEFINE_SCALED_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                partial_bmm, at::_ops::bmm(),
                utils::_ops::fill_identity_zeros(), impl::_ops::mm_scale())

        PT_DEFINE_SCALED_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                partial_matmul, at::_ops::matmul(),
                utils::_ops::fill_identity_zeros(), impl::_ops::mm_scale())

        PT_DEFINE_SCALED_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                linalg_partial_matmul, at::_ops::linalg_matmul(),
                utils::_ops::fill_identity_zeros(), impl::_ops::mm_scale())

        PT_DEFINE_SCALED_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                partial_mv, at::_ops::mv(),
                utils::_ops::fill_identity_zeros(), impl::_ops::mm_scale())

        PT_DEFINE_SCALED_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                partial_inner, at::_ops::inner(),
                utils::_ops::fill_identity_zeros(), impl::_ops::mm_scale())

        PT_DEFINE_SCALED_BINARY_OPS_FORALL_TENSOR_OVERLOADS(
                partial_outer, at::_ops::outer(),
                utils::_ops::fill_identity_zeros(), 1)

        TORCH_LIBRARY_FRAGMENT(partialtorch, m) {
            // arithmetics
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS(partial_ger,)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS(partial_mm,)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS(partial_bmm,)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS(partial_matmul,)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS(linalg_partial_matmul,)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS(partial_mv,)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS(partial_inner,)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS(partial_outer,)

            // scaled arithmetics
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    partial_ger, scaled,
                    bool, scaled,)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    partial_mm, scaled,
                    bool, scaled,)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    partial_bmm, scaled,
                    bool, scaled,)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    partial_matmul, scaled,
                    bool, scaled,)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    linalg_partial_matmul, scaled,
                    bool, scaled,)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    partial_mv, scaled,
                    bool, scaled,)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    partial_inner, scaled,
                    bool, scaled,)
            PT_REGISTER_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                    partial_outer, scaled,
                    bool, scaled,)
        }
    }
}
