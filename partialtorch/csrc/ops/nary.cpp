#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <torch/library.h>

#include "../MaskedPair.h"
#include "izero_div.h"
#include "utils/fill_identity.h"
#include "utils/mask_utils.h"
#include "utils/partial_rescaling_utils.h"
#include "utils/schema_utils.h"

// ~~~~~ ops definition macros ~~~~~
#define PT_DEFINE_NARY_OP(NAME, IMPL_OP, IDENTITY_OP, INPUT_T)                     \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                         \
    INPUT_T tensors) {                                                             \
    static constexpr auto op = IMPL_OP;                                            \
    static constexpr auto fill_identity_op = IDENTITY_OP;                          \
    return impl::partial_nary_impl<false>(op, fill_identity_op, nullptr, tensors); \
}

#define PT_DEFINE_NARY_OP_WITH(NAME, IMPL_OP, IDENTITY_OP, INPUT_T, ARG1_T, ARG1_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                             \
    INPUT_T tensors,                                                                   \
    ARG1_T ARG1_NAME) {                                                                \
    static constexpr auto op = IMPL_OP;                                                \
    static constexpr auto fill_identity_op = IDENTITY_OP;                              \
    return impl::partial_nary_impl<false>(op, fill_identity_op, nullptr, tensors, ARG1_NAME); \
}

#define PT_DEFINE_NARY_OP_WITH2(NAME, IMPL_OP, IDENTITY_OP, INPUT_T, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                                 \
    INPUT_T tensors,                                                                                       \
    ARG1_T ARG1_NAME,                                                                                      \
    ARG2_T ARG2_NAME) {                                                                                    \
    static constexpr auto op = IMPL_OP;                                                                    \
    static constexpr auto fill_identity_op = IDENTITY_OP;                                                  \
    return impl::partial_nary_impl<false>(op, fill_identity_op, nullptr, tensors, ARG1_NAME, ARG2_NAME);   \
}

#define PT_DEFINE_NARY_OPS_FORALL_TENSOR_OVERLOADS(NAME, TENSOR_OP, IDENTITY_OP) \
PT_DEFINE_NARY_OP(NAME, TENSOR_OP, IDENTITY_OP, TensorMaskedPairIntrusivePtrList)\
PT_DEFINE_NARY_OP(NAME, TENSOR_OP, IDENTITY_OP, at::TensorList)

#define PT_DEFINE_NARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, TENSOR_OP, IDENTITY_OP, ARG1_T, ARG1_NAME) \
PT_DEFINE_NARY_OP_WITH(NAME, TENSOR_OP, IDENTITY_OP, TensorMaskedPairIntrusivePtrList, ARG1_T, ARG1_NAME)\
PT_DEFINE_NARY_OP_WITH(NAME, TENSOR_OP, IDENTITY_OP, at::TensorList, ARG1_T, ARG1_NAME)

#define PT_DEFINE_NARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, TENSOR_OP, IDENTITY_OP, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME) \
PT_DEFINE_NARY_OP_WITH2(NAME, TENSOR_OP, IDENTITY_OP, TensorMaskedPairIntrusivePtrList, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)\
PT_DEFINE_NARY_OP_WITH2(NAME, TENSOR_OP, IDENTITY_OP, at::TensorList, ARG1_T, ARG1_NAME, ARG2_T, ARG2_NAME)

// scaled ops
#define PT_DEFINE_SCALED_NARY_OP(NAME, IMPL_OP, IDENTITY_OP, CANONICAL_SCALE, INPUT_T) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                             \
    INPUT_T tensors,                                                                   \
    bool scaled) {                                                                     \
    static constexpr auto op = IMPL_OP;                                                \
    static constexpr auto fill_identity_op = IDENTITY_OP;                              \
    static constexpr auto canonical_scale = CANONICAL_SCALE;                           \
    if (!scaled)                                                                       \
        return impl::partial_nary_impl<false>(op, fill_identity_op, nullptr, tensors); \
    return impl::partial_nary_impl<true>(op, fill_identity_op, canonical_scale, tensors); \
}

#define PT_DEFINE_SCALED_NARY_OP_WITH(NAME, IMPL_OP, IDENTITY_OP, CANONICAL_SCALE, INPUT_T, ARG1_T, ARG1_NAME) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                                                     \
    INPUT_T tensors,                                                                                           \
    ARG1_T ARG1_NAME,                                                                                          \
    bool scaled) {                                                                                             \
    static constexpr auto op = IMPL_OP;                                                                        \
    static constexpr auto fill_identity_op = IDENTITY_OP;                                                      \
    static constexpr auto canonical_scale = CANONICAL_SCALE;                                                   \
    if (!scaled)                                                                                               \
        return impl::partial_nary_impl<false>(op, fill_identity_op, nullptr, tensors, ARG1_NAME);              \
    return impl::partial_nary_impl<true>(op, fill_identity_op, canonical_scale, tensors, ARG1_NAME);           \
}

#define PT_DEFINE_SCALED_NARY_OPS_FORALL_TENSOR_OVERLOADS(NAME, TENSOR_OP, IDENTITY_OP, CANONICAL_SCALE) \
PT_DEFINE_SCALED_NARY_OP(NAME, TENSOR_OP, IDENTITY_OP, CANONICAL_SCALE, TensorMaskedPairIntrusivePtrList)\
PT_DEFINE_SCALED_NARY_OP(NAME, TENSOR_OP, IDENTITY_OP, CANONICAL_SCALE, at::TensorList, const at::Tensor &)

#define PT_DEFINE_SCALED_NARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, TENSOR_OP, IDENTITY_OP, CANONICAL_SCALE, ARG1_T, ARG1_NAME) \
PT_DEFINE_SCALED_NARY_OP_WITH(NAME, TENSOR_OP, IDENTITY_OP, CANONICAL_SCALE, TensorMaskedPairIntrusivePtrList, ARG1_T, ARG1_NAME)\
PT_DEFINE_SCALED_NARY_OP_WITH(NAME, TENSOR_OP, IDENTITY_OP, CANONICAL_SCALE, at::TensorList, const at::Tensor &, ARG1_T, ARG1_NAME)

// ~~~~~ ops registration macros ~~~~~
#define PT_NARY_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, INPUT_T) \
utils::FunctionSchemaBuilder(#NAME).add_overload(#OVERLOAD_NAME).add_overload(#POSTFIX).arg<INPUT_T>("tensors").ret<TensorMaskedPair>()

#define PT_REGISTER_NARY_OP(NAME, OVERLOAD_NAME, POSTFIX, INPUT_T) \
m.def(PT_NARY_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, INPUT_T).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(INPUT_T)>(NAME)));

#define PT_REGISTER_NARY_OP_WITH(NAME, OVERLOAD_NAME, POSTFIX, INPUT_T, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
m.def(PT_NARY_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, INPUT_T).vararg().arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(INPUT_T, ARG1_T)>(NAME)));

#define PT_REGISTER_NARY_OP_WITH2(NAME, OVERLOAD_NAME, POSTFIX, INPUT_T, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
m.def(PT_NARY_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, INPUT_T).vararg(VARARG_INDEX).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).arg<ARG2_T>(#ARG2_NAME, #ARG2_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(INPUT_T, ARG1_T, ARG2_T)>(NAME)));

#define PT_REGISTER_NARY_OP_WITH3(NAME, OVERLOAD_NAME, POSTFIX, INPUT_T, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT) \
m.def(PT_NARY_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, INPUT_T).vararg(VARARG_INDEX).arg<ARG1_T>(#ARG1_NAME, #ARG1_DEFAULT).arg<ARG2_T>(#ARG2_NAME, #ARG2_DEFAULT).arg<ARG3_T>(#ARG3_NAME, #ARG3_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(INPUT_T, ARG1_T, ARG2_T, ARG3_T)>(NAME)));

#define PT_REGISTER_NARY_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX) \
PT_REGISTER_NARY_OP(NAME, MaskedPairList, POSTFIX, TensorMaskedPairIntrusivePtrList) \
PT_REGISTER_NARY_OP(NAME, TensorList, POSTFIX, at::TensorList)

#define PT_REGISTER_NARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, POSTFIX, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_NARY_OP_WITH(NAME, MaskedPairList, POSTFIX, TensorMaskedPairIntrusivePtrList, ARG1_T, ARG1_NAME, ARG1_DEFAULT) \
PT_REGISTER_NARY_OP_WITH(NAME, TensorList, POSTFIX, at::TensorList, ARG1_T, ARG1_NAME, ARG1_DEFAULT)

#define PT_REGISTER_NARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, POSTFIX, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_NARY_OP_WITH2(NAME, MaskedPairList, POSTFIX, TensorMaskedPairIntrusivePtrList, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT) \
PT_REGISTER_NARY_OP_WITH2(NAME, TensorList, POSTFIX, at::TensorList, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT)

#define PT_REGISTER_NARY_OPS_FORALL_TENSOR_OVERLOADS_WITH3(NAME, POSTFIX, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT) \
PT_REGISTER_NARY_OP_WITH3(NAME, MaskedPairList, POSTFIX, TensorMaskedPairIntrusivePtrList, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT) \
PT_REGISTER_NARY_OP_WITH3(NAME, TensorList, POSTFIX, at::TensorList, VARARG_INDEX, ARG1_T, ARG1_NAME, ARG1_DEFAULT, ARG2_T, ARG2_NAME, ARG2_DEFAULT, ARG3_T, ARG3_NAME, ARG3_DEFAULT)

namespace partialtorch {
    namespace ops {
        namespace impl {
            template<bool scaled = false, typename op_T, typename fill_identity_op_T, typename canonical_scale_T,
                    typename input_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> partial_nary_impl(
                    op_T &&op,
                    fill_identity_op_T &&fill_identity_op,
                    canonical_scale_T &&canonical_scale,
                    at::ArrayRef<input_T> inputs,
                    Args &&... args) {
                if (!utils::has_tensor_mask(inputs))
                    return masked_pair(op.call(utils::get_data(inputs), args...));

                auto output_data = op.call(fill_identity_op.call(inputs), args...);
                auto mask_ratio_options = output_data.options();
                at::Tensor mask_ratio;
                {
                    at::NoGradGuard g;
                    mask_ratio = op.call(
                            utils::get_tensor_mask(inputs, mask_ratio_options), args...);
                    if constexpr (scaled) {
                        if constexpr (std::is_same_v<std::base_t<canonical_scale_T>, nullptr_t>) {
                            utils::izero_ldiv_(
                                    mask_ratio,
                                    op.call(utils::ones_like(inputs), args...)
                            );
                        } else if constexpr (std::is_same_v<std::base_t<canonical_scale_T>, at::Scalar> ||
                                             std::is_same_v<std::base_t<canonical_scale_T>, at::Tensor> ||
                                             std::is_arithmetic_v<std::base_t<canonical_scale_T>>) {
                            utils::izero_ldiv_(mask_ratio, canonical_scale);
                        } else {
                            utils::izero_ldiv_(
                                    mask_ratio,
                                    canonical_scale.call(utils::get_data(inputs), args...)
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

            template<typename input_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> partial_einsum_impl(
                    c10::string_view equation,
                    at::ArrayRef<input_T> operands,
                    Args &&... args) {
                if (!utils::has_tensor_mask(operands))
                    return masked_pair(at::einsum(equation, utils::get_data(operands), args...));

                auto output_data = at::einsum(
                        equation, utils::_ops::fill_identity_zeros<false>::call(operands), args...);
                auto mask_options = output_data.scalar_type();
                at::Tensor output_mask;
                {
                    at::NoGradGuard g;
                    output_mask = at::einsum(equation, utils::get_tensor_mask(operands, mask_options), args...);
                }
                output_mask = output_mask.to(at::kBool);
                return masked_pair(output_data, output_mask);
            }
        }

        PT_DEFINE_NARY_OPS_FORALL_TENSOR_OVERLOADS(
                linalg_partial_multi_dot, at::_ops::linalg_multi_dot(), utils::_ops::fill_identity_zeros())

        c10::intrusive_ptr<TensorMaskedPair> partial_einsum(
                c10::string_view equation, TensorMaskedPairIntrusivePtrList tensors, at::OptionalIntArrayRef path) {
            return impl::partial_einsum_impl(equation, tensors, path);
        }

        c10::intrusive_ptr<TensorMaskedPair> partial_einsum(
                c10::string_view equation, at::TensorList tensors, at::OptionalIntArrayRef path) {
            return impl::partial_einsum_impl(equation, tensors, path);
        }

        TORCH_LIBRARY_FRAGMENT(partialtorch, m) {
            PT_REGISTER_NARY_OPS_FORALL_TENSOR_OVERLOADS(linalg_partial_multi_dot,)

            m.def(utils::FunctionSchemaBuilder("partial_einsum").overload("MaskedPairList")
                          .arg<c10::string_view>("equation")
                          .arg<TensorMaskedPairIntrusivePtrList>("tensors")
                          .vararg().arg<at::OptionalIntArrayRef>("path", "None")
                          .ret<c10::intrusive_ptr<TensorMaskedPair>>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          c10::string_view,
                          TensorMaskedPairIntrusivePtrList,
                          at::OptionalIntArrayRef)>(partial_einsum)));
            m.def(utils::FunctionSchemaBuilder("partial_einsum").overload("TensorList")
                          .arg<c10::string_view>("equation")
                          .arg<at::TensorList>("tensors")
                          .vararg().arg<at::OptionalIntArrayRef>("path", "None")
                          .ret<c10::intrusive_ptr<TensorMaskedPair>>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          c10::string_view,
                          at::TensorList,
                          at::OptionalIntArrayRef)>(partial_einsum)));
        }
    }
}
