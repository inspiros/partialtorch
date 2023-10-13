#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <torch/library.h>

#include "../MaskedPair.h"
#include "reduction.h"
#include "utils/fill_identity.h"
#include "utils/irepeat.h"
#include "utils/result_with_indices.h"
#include "utils/mask_utils.h"
#include "utils/schema_utils.h"

// ~~~~~ ops definition macros ~~~~~
#define PT_DEFINE_MAX_POOLND_OP(NAME, IMPL_OP, MASK_OP, INPUT_T) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                       \
        INPUT_T input,                                           \
        at::IntArrayRef kernel_size,                             \
        at::IntArrayRef stride,                                  \
        at::IntArrayRef padding,                                 \
        at::IntArrayRef dilation,                                \
        bool ceil_mode) {                                        \
    static constexpr auto op = IMPL_OP;                          \
    static constexpr auto mask_op = MASK_OP;                     \
    return impl::max_poolnd_impl(                                \
            op, mask_op, input, kernel_size, stride, padding, dilation, ceil_mode); \
}

#define PT_DEFINE_MAX_POOLND_WITH_INDICES_OP(NAME, IMPL_OP, MASK_OP, INPUT_T) \
std::tuple<c10::intrusive_ptr<TensorMaskedPair>, at::Tensor> NAME(            \
        INPUT_T input,                                                        \
        at::IntArrayRef kernel_size,                                          \
        at::IntArrayRef stride,                                               \
        at::IntArrayRef padding,                                              \
        at::IntArrayRef dilation,                                             \
        bool ceil_mode) {                                                     \
    static constexpr auto op = IMPL_OP;                                       \
    static constexpr auto mask_op = MASK_OP;                                  \
    return impl::max_poolnd_with_indices_impl(                                \
            op, mask_op, input, kernel_size, stride, padding, dilation, ceil_mode); \
}

#define PT_DEFINE_FRACTIONAL_MAX_POOLND_OP(NAME, IMPL_OP, MASK_OP, INPUT_T) \
std::tuple<c10::intrusive_ptr<TensorMaskedPair>, at::Tensor> NAME(          \
        INPUT_T input,                                                      \
        at::IntArrayRef kernel_size,                                        \
        at::IntArrayRef output_size,                                        \
        const at::Tensor &random_samples) {                                 \
    static constexpr auto op = IMPL_OP;                                     \
    static constexpr auto mask_op = MASK_OP;                                \
    return impl::max_poolnd_with_indices_impl(                              \
            op, mask_op, input, kernel_size, output_size, random_samples);  \
}

#define PT_DEFINE_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, MASK_OP) \
PT_DEFINE_MAX_POOLND_OP(NAME, IMPL_OP, MASK_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DEFINE_MAX_POOLND_OP(NAME, IMPL_OP, MASK_OP, const at::Tensor &)

#define PT_DEFINE_MAX_POOLND_WITH_INDICES_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, MASK_OP) \
PT_DEFINE_MAX_POOLND_WITH_INDICES_OP(NAME, IMPL_OP, MASK_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DEFINE_MAX_POOLND_WITH_INDICES_OP(NAME, IMPL_OP, MASK_OP, const at::Tensor &)

#define PT_DEFINE_FRACTIONAL_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, MASK_OP) \
PT_DEFINE_FRACTIONAL_MAX_POOLND_OP(NAME, IMPL_OP, MASK_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DEFINE_FRACTIONAL_MAX_POOLND_OP(NAME, IMPL_OP, MASK_OP, const at::Tensor &)

// ~~~~~ ops registration macros ~~~~~
#define PT_MAX_POOLND_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T) \
utils::FunctionSchemaBuilder(#NAME).add_overload(#OVERLOAD_NAME).add_overload(#POSTFIX).arg<INPUT_T>("input").arg<at::IntArrayRef>("kernel_size").arg<at::IntArrayRef, DIMENSION>("stride", "[]").arg<at::IntArrayRef, DIMENSION>("padding", "0").arg<at::IntArrayRef, DIMENSION>("dilation", "1").arg<bool>("ceil_mode", "False").ret<TensorMaskedPair>()

#define PT_MAX_POOLND_WITH_INDICES_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T) \
utils::FunctionSchemaBuilder(#NAME).add_overload(#OVERLOAD_NAME).add_overload(#POSTFIX).arg<INPUT_T>("input").arg<at::IntArrayRef>("kernel_size").arg<at::IntArrayRef, DIMENSION>("stride", "[]").arg<at::IntArrayRef, DIMENSION>("padding", "0").arg<at::IntArrayRef, DIMENSION>("dilation", "1").arg<bool>("ceil_mode", "False").ret<TensorMaskedPair>().ret<at::Tensor>()

#define PT_FRACTIONAL_MAX_POOLND_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T) \
utils::FunctionSchemaBuilder(#NAME).add_overload(#OVERLOAD_NAME).add_overload(#POSTFIX).arg<INPUT_T>("input").arg<at::IntArrayRef>("kernel_size").arg<at::IntArrayRef, DIMENSION>("output_size").arg<const at::Tensor &>("random_samples").ret<TensorMaskedPair>().ret<at::Tensor>()

#define PT_REGISTER_MAX_POOLND_OP(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T) \
m.def(PT_MAX_POOLND_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(INPUT_T, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, bool)>(NAME)));

#define PT_REGISTER_MAX_POOLND_WITH_INDICES_OP(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T) \
m.def(PT_MAX_POOLND_WITH_INDICES_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T).schema().c_str(), \
    TORCH_FN(static_cast<masked_pair_with_indices (*)(INPUT_T, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, bool)>(NAME)));

#define PT_REGISTER_FRACTIONAL_MAX_POOLND_OP(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T) \
m.def(PT_FRACTIONAL_MAX_POOLND_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T).schema().c_str(), \
    TORCH_FN(static_cast<masked_pair_with_indices (*)(INPUT_T, at::IntArrayRef, at::IntArrayRef, const at::Tensor &)>(NAME)));

#define PT_REGISTER_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, DIMENSION) \
PT_REGISTER_MAX_POOLND_OP(NAME, MaskedPair, POSTFIX, DIMENSION, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_REGISTER_MAX_POOLND_OP(NAME, Tensor, POSTFIX, DIMENSION, const at::Tensor &)

#define PT_REGISTER_MAX_POOLND_WITH_INDICES_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, DIMENSION) \
PT_REGISTER_MAX_POOLND_WITH_INDICES_OP(NAME, MaskedPair, POSTFIX, DIMENSION, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_REGISTER_MAX_POOLND_WITH_INDICES_OP(NAME, Tensor, POSTFIX, DIMENSION, const at::Tensor &)

#define PT_REGISTER_FRACTIONAL_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, DIMENSION) \
PT_REGISTER_FRACTIONAL_MAX_POOLND_OP(NAME, MaskedPair, POSTFIX, DIMENSION, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_REGISTER_FRACTIONAL_MAX_POOLND_OP(NAME, Tensor, POSTFIX, DIMENSION, const at::Tensor &)

namespace partialtorch {
    namespace ops {
        namespace impl {
            template<typename op_T, typename mask_op_T, typename input_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> max_poolnd_impl(
                    op_T &&op,
                    mask_op_T &&mask_op,
                    const input_T &input,
                    Args &&... args) {
                if (!utils::has_tensor_mask(input))
                    return masked_pair(op.call(utils::get_data(input), args...));

                auto output_data = op.call(
                        utils::_ops::fill_identity_neginf<false>::call(input), args...);
                at::Tensor output_mask;
                if constexpr (std::is_same_v<std::base_t<mask_op_T>, nullptr_t>) {
                    {
                        at::NoGradGuard g;
                        output_mask = op.call(
                                utils::get_tensor_mask(input, output_data.options()), args...);
                    }
                    output_mask = output_mask.to(at::kBool);
                } else {
                    output_mask = mask_op.call(
                            utils::get_tensor_mask(input), args...);
                }
                return masked_pair(output_data, output_mask);
            }

            template<typename op_T, typename mask_op_T, typename input_T, typename... Args>
            static C10_ALWAYS_INLINE std::tuple<c10::intrusive_ptr<TensorMaskedPair>, at::Tensor>
            max_poolnd_with_indices_impl(
                    op_T &&op,
                    mask_op_T &&mask_op,
                    const input_T &input,
                    Args &&... args) {
                if (!utils::has_tensor_mask(input)) {
                    auto result = op.call(utils::get_data(input), args...);
                    auto output_data = std::get<0>(result);
                    auto output_indices = std::get<1>(result);
                    return std::make_tuple(masked_pair(output_data), output_indices);
                }

                auto result = op.call(
                        utils::_ops::fill_identity_neginf<false>::call(input), args...);
                auto output_data = std::get<0>(result);
                auto output_indices = std::get<1>(result);
                at::Tensor output_mask;
                if constexpr (std::is_same_v<std::base_t<mask_op_T>, nullptr_t>) {
                    {
                        at::NoGradGuard g;
                        output_mask = std::get<0>(op.call(
                                utils::get_tensor_mask(input, output_data.options()), args...));
                    }
                    output_mask = output_mask.to(at::kBool);
                } else {
                    output_mask = mask_op.call(
                            utils::get_tensor_mask(input), args...);
                }
                output_indices.masked_fill_(output_mask.logical_not(), -1);
                return std::make_tuple(masked_pair(output_data, output_mask), output_indices);
            }
        }

        // TODO: implement any_poolnd for mask_ops
        // max_pool
        PT_DEFINE_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(
                max_pool1d, at::_ops::max_pool1d(), nullptr)

        PT_DEFINE_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(
                max_pool2d, at::_ops::max_pool2d(), nullptr)

        PT_DEFINE_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(
                max_pool3d, at::_ops::max_pool3d(), nullptr)

        PT_DEFINE_MAX_POOLND_WITH_INDICES_OPS_FORALL_TENSOR_OVERLOADS(
                max_pool1d_with_indices, at::_ops::max_pool1d_with_indices(), nullptr)

        PT_DEFINE_MAX_POOLND_WITH_INDICES_OPS_FORALL_TENSOR_OVERLOADS(
                max_pool2d_with_indices, at::_ops::max_pool2d_with_indices(), nullptr)

        PT_DEFINE_MAX_POOLND_WITH_INDICES_OPS_FORALL_TENSOR_OVERLOADS(
                max_pool3d_with_indices, at::_ops::max_pool3d_with_indices(), nullptr)

        // fractional_max_pool
        PT_DEFINE_FRACTIONAL_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(
                fractional_max_pool2d, at::_ops::fractional_max_pool2d(), nullptr)

        PT_DEFINE_FRACTIONAL_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(
                fractional_max_pool3d, at::_ops::fractional_max_pool3d(), nullptr)

        TORCH_LIBRARY_FRAGMENT(partialtorch, m) {
            // max_pool
            PT_REGISTER_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(max_pool1d, , 1)
            PT_REGISTER_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(max_pool2d, , 2)
            PT_REGISTER_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(max_pool3d, , 3)
            PT_REGISTER_MAX_POOLND_WITH_INDICES_OPS_FORALL_TENSOR_OVERLOADS(max_pool1d_with_indices, , 1)
            PT_REGISTER_MAX_POOLND_WITH_INDICES_OPS_FORALL_TENSOR_OVERLOADS(max_pool2d_with_indices, , 2)
            PT_REGISTER_MAX_POOLND_WITH_INDICES_OPS_FORALL_TENSOR_OVERLOADS(max_pool3d_with_indices, , 3)

            // fractional_max_pool
            PT_REGISTER_FRACTIONAL_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(fractional_max_pool2d, , 2)
            PT_REGISTER_FRACTIONAL_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(fractional_max_pool3d, , 3)
        }
    }
}
