#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <torch/library.h>

#include "../MaskedPair.h"
#include "masked_adaptive_avg_pool.h"
#include "masked_avg_pool.h"
#include "reduction.h"
#include "utils/fill_identity.h"
#include "utils/irepeat.h"
#include "utils/result_with_indices.h"
#include "utils/mask_utils.h"
#include "utils/schema_utils.h"

// ~~~~~ ops definition macros ~~~~~
#define PT_DEFINE_AVG_POOLND_OP(NAME, ATEN_OP, IMPL_OP, INPUT_T) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                       \
        INPUT_T input,                                           \
        at::IntArrayRef kernel_size,                             \
        at::IntArrayRef stride,                                  \
        at::IntArrayRef padding,                                 \
        bool ceil_mode,                                          \
        bool count_include_pad) {                                \
    static constexpr auto aten_op = ATEN_OP;                     \
    static constexpr auto impl_op = IMPL_OP;                     \
    return impl::avg_poolnd_impl(                                \
            aten_op, impl_op, input, kernel_size, stride, padding, ceil_mode, count_include_pad); \
}

#define PT_DEFINE_ADAPTIVE_AVG_POOLND_OP(NAME, ATEN_OP, IMPL_OP, INPUT_T, OUTPUT_SIZE_T) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                                               \
        INPUT_T input,                                                                   \
        OUTPUT_SIZE_T output_size) {                                                     \
    static constexpr auto aten_op = ATEN_OP;                                             \
    static constexpr auto impl_op = IMPL_OP;                                             \
    return impl::avg_poolnd_impl(                                                        \
            aten_op, impl_op, input, output_size);                                       \
}

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

#define PT_DEFINE_ADAPTIVE_MAX_POOLND_WITH_INDICES_OP(NAME, IMPL_OP, MASK_OP, INPUT_T) \
std::tuple<c10::intrusive_ptr<TensorMaskedPair>, at::Tensor> NAME(                     \
        INPUT_T input,                                                                 \
        at::IntArrayRef output_size) {                                                 \
    static constexpr auto op = IMPL_OP;                                                \
    static constexpr auto mask_op = MASK_OP;                                           \
    return impl::max_poolnd_with_indices_impl(                                         \
            op, mask_op, input, output_size);                                          \
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

#define PT_DEFINE_AVG_POOLND_OPS_FORALL_TENSOR_OVERLOADS(NAME, ATEN_OP, IMPL_OP) \
PT_DEFINE_AVG_POOLND_OP(NAME, ATEN_OP, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DEFINE_AVG_POOLND_OP(NAME, ATEN_OP, IMPL_OP, const at::Tensor &)

#define PT_DEFINE_ADAPTIVE_AVG_POOLND_OPS_FORALL_TENSOR_OVERLOADS(NAME, ATEN_OP, IMPL_OP, OUTPUT_SIZE_T) \
PT_DEFINE_ADAPTIVE_AVG_POOLND_OP(NAME, ATEN_OP, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, OUTPUT_SIZE_T) \
PT_DEFINE_ADAPTIVE_AVG_POOLND_OP(NAME, ATEN_OP, IMPL_OP, const at::Tensor &, OUTPUT_SIZE_T)

#define PT_DEFINE_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, MASK_OP) \
PT_DEFINE_MAX_POOLND_OP(NAME, IMPL_OP, MASK_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DEFINE_MAX_POOLND_OP(NAME, IMPL_OP, MASK_OP, const at::Tensor &)

#define PT_DEFINE_MAX_POOLND_WITH_INDICES_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, MASK_OP) \
PT_DEFINE_MAX_POOLND_WITH_INDICES_OP(NAME, IMPL_OP, MASK_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DEFINE_MAX_POOLND_WITH_INDICES_OP(NAME, IMPL_OP, MASK_OP, const at::Tensor &)

#define PT_DEFINE_ADAPTIVE_MAX_POOLND_WITH_INDICES_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, MASK_OP) \
PT_DEFINE_ADAPTIVE_MAX_POOLND_WITH_INDICES_OP(NAME, IMPL_OP, MASK_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DEFINE_ADAPTIVE_MAX_POOLND_WITH_INDICES_OP(NAME, IMPL_OP, MASK_OP, const at::Tensor &)

#define PT_DEFINE_FRACTIONAL_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, MASK_OP) \
PT_DEFINE_FRACTIONAL_MAX_POOLND_OP(NAME, IMPL_OP, MASK_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DEFINE_FRACTIONAL_MAX_POOLND_OP(NAME, IMPL_OP, MASK_OP, const at::Tensor &)

// ~~~~~ ops registration macros ~~~~~
#define PT_AVG_POOLND_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T) \
utils::FunctionSchemaBuilder(#NAME).add_overload(#OVERLOAD_NAME).add_overload(#POSTFIX).arg<INPUT_T>("input").arg<at::IntArrayRef, DIMENSION>("kernel_size").arg<at::IntArrayRef, DIMENSION>("stride", "[]").arg<at::IntArrayRef, DIMENSION>("padding", "0").arg<bool>("ceil_mode", "False").arg<bool>("count_include_pad", "True").ret<TensorMaskedPair>()

#define PT_ADAPTIVE_AVG_POOLND_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T, OUTPUT_SIZE_T) \
utils::FunctionSchemaBuilder(#NAME).add_overload(#OVERLOAD_NAME).add_overload(#POSTFIX).arg<INPUT_T>("input").arg<OUTPUT_SIZE_T, DIMENSION>("output_size").ret<TensorMaskedPair>()

#define PT_MAX_POOLND_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T) \
utils::FunctionSchemaBuilder(#NAME).add_overload(#OVERLOAD_NAME).add_overload(#POSTFIX).arg<INPUT_T>("input").arg<at::IntArrayRef, DIMENSION>("kernel_size").arg<at::IntArrayRef, DIMENSION>("stride", "[]").arg<at::IntArrayRef, DIMENSION>("padding", "0").arg<at::IntArrayRef, DIMENSION>("dilation", "1").arg<bool>("ceil_mode", "False").ret<TensorMaskedPair>()

#define PT_MAX_POOLND_WITH_INDICES_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T) \
utils::FunctionSchemaBuilder(#NAME).add_overload(#OVERLOAD_NAME).add_overload(#POSTFIX).arg<INPUT_T>("input").arg<at::IntArrayRef, DIMENSION>("kernel_size").arg<at::IntArrayRef, DIMENSION>("stride", "[]").arg<at::IntArrayRef, DIMENSION>("padding", "0").arg<at::IntArrayRef, DIMENSION>("dilation", "1").arg<bool>("ceil_mode", "False").ret<TensorMaskedPair>().ret<at::Tensor>()

#define PT_ADAPTIVE_MAX_POOLND_WITH_INDICES_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T) \
utils::FunctionSchemaBuilder(#NAME).add_overload(#OVERLOAD_NAME).add_overload(#POSTFIX).arg<INPUT_T>("input").arg<at::IntArrayRef, DIMENSION>("output_size").ret<TensorMaskedPair>().ret<at::Tensor>()

#define PT_FRACTIONAL_MAX_POOLND_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T) \
utils::FunctionSchemaBuilder(#NAME).add_overload(#OVERLOAD_NAME).add_overload(#POSTFIX).arg<INPUT_T>("input").arg<at::IntArrayRef, DIMENSION>("kernel_size").arg<at::IntArrayRef, DIMENSION>("output_size").arg<const at::Tensor &>("random_samples").ret<TensorMaskedPair>().ret<at::Tensor>()

#define PT_REGISTER_AVG_POOLND_OP(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T) \
m.def(PT_AVG_POOLND_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(INPUT_T, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, bool, bool)>(NAME)));

#define PT_REGISTER_ADAPTIVE_AVG_POOLND_OP(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T, OUTPUT_SIZE_T) \
m.def(PT_ADAPTIVE_AVG_POOLND_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T, OUTPUT_SIZE_T).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(INPUT_T, OUTPUT_SIZE_T)>(NAME)));

#define PT_REGISTER_MAX_POOLND_OP(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T) \
m.def(PT_MAX_POOLND_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(INPUT_T, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, bool)>(NAME)));

#define PT_REGISTER_MAX_POOLND_WITH_INDICES_OP(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T) \
m.def(PT_MAX_POOLND_WITH_INDICES_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T).schema().c_str(), \
    TORCH_FN(static_cast<masked_pair_with_indices (*)(INPUT_T, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, bool)>(NAME)));

#define PT_REGISTER_ADAPTIVE_MAX_POOLND_WITH_INDICES_OP(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T) \
m.def(PT_ADAPTIVE_MAX_POOLND_WITH_INDICES_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T).schema().c_str(), \
    TORCH_FN(static_cast<masked_pair_with_indices (*)(INPUT_T, at::IntArrayRef)>(NAME)));

#define PT_REGISTER_FRACTIONAL_MAX_POOLND_OP(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T) \
m.def(PT_FRACTIONAL_MAX_POOLND_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T).schema().c_str(), \
    TORCH_FN(static_cast<masked_pair_with_indices (*)(INPUT_T, at::IntArrayRef, at::IntArrayRef, const at::Tensor &)>(NAME)));

#define PT_REGISTER_AVG_POOLND_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, DIMENSION) \
PT_REGISTER_AVG_POOLND_OP(NAME, MaskedPair, POSTFIX, DIMENSION, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_REGISTER_AVG_POOLND_OP(NAME, Tensor, POSTFIX, DIMENSION, const at::Tensor &)

#define PT_REGISTER_ADAPTIVE_AVG_POOLND_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, DIMENSION, OUTPUT_SIZE_T) \
PT_REGISTER_ADAPTIVE_AVG_POOLND_OP(NAME, MaskedPair, POSTFIX, DIMENSION, const_intrusive_ptr_arg_t<TensorMaskedPair>, OUTPUT_SIZE_T) \
PT_REGISTER_ADAPTIVE_AVG_POOLND_OP(NAME, Tensor, POSTFIX, DIMENSION, const at::Tensor &, OUTPUT_SIZE_T)

#define PT_REGISTER_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, DIMENSION) \
PT_REGISTER_MAX_POOLND_OP(NAME, MaskedPair, POSTFIX, DIMENSION, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_REGISTER_MAX_POOLND_OP(NAME, Tensor, POSTFIX, DIMENSION, const at::Tensor &)

#define PT_REGISTER_MAX_POOLND_WITH_INDICES_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, DIMENSION) \
PT_REGISTER_MAX_POOLND_WITH_INDICES_OP(NAME, MaskedPair, POSTFIX, DIMENSION, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_REGISTER_MAX_POOLND_WITH_INDICES_OP(NAME, Tensor, POSTFIX, DIMENSION, const at::Tensor &)

#define PT_REGISTER_ADAPTIVE_MAX_POOLND_WITH_INDICES_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, DIMENSION) \
PT_REGISTER_ADAPTIVE_MAX_POOLND_WITH_INDICES_OP(NAME, MaskedPair, POSTFIX, DIMENSION, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_REGISTER_ADAPTIVE_MAX_POOLND_WITH_INDICES_OP(NAME, Tensor, POSTFIX, DIMENSION, const at::Tensor &)

#define PT_REGISTER_FRACTIONAL_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, DIMENSION) \
PT_REGISTER_FRACTIONAL_MAX_POOLND_OP(NAME, MaskedPair, POSTFIX, DIMENSION, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_REGISTER_FRACTIONAL_MAX_POOLND_OP(NAME, Tensor, POSTFIX, DIMENSION, const at::Tensor &)

namespace partialtorch {
    namespace ops {
        namespace impl {
            namespace _ops {
                struct avg_pool2d_no_divisor_override {
                    static inline at::Tensor call(
                            const at::Tensor &data,
                            at::IntArrayRef kernel_size,
                            at::IntArrayRef stride,
                            at::IntArrayRef padding,
                            bool ceil_mode,
                            bool count_include_pad) {
                        return at::_ops::avg_pool2d::call(data, kernel_size, stride, padding,
                                                          ceil_mode, count_include_pad, {});
                    }
                };

                struct avg_pool3d_no_divisor_override {
                    static inline at::Tensor call(
                            const at::Tensor &data,
                            at::IntArrayRef kernel_size,
                            at::IntArrayRef stride,
                            at::IntArrayRef padding,
                            bool ceil_mode,
                            bool count_include_pad) {
                        return at::_ops::avg_pool3d::call(data, kernel_size, stride, padding,
                                                          ceil_mode, count_include_pad, {});
                    }
                };

                struct masked_avg_pool1d {
                    static inline std::tuple<at::Tensor, at::Tensor> call(
                            const at::Tensor &data,
                            const at::Tensor &mask,
                            at::IntArrayRef kernel_size,
                            at::IntArrayRef stride,
                            at::IntArrayRef padding,
                            bool ceil_mode,
                            bool count_include_pad) {
                        return _masked_avg_pool1d(data, mask, kernel_size, stride, padding,
                                                  ceil_mode, count_include_pad);
                    }
                };

                struct masked_avg_pool2d {
                    static inline std::tuple<at::Tensor, at::Tensor> call(
                            const at::Tensor &data,
                            const at::Tensor &mask,
                            at::IntArrayRef kernel_size,
                            at::IntArrayRef stride,
                            at::IntArrayRef padding,
                            bool ceil_mode,
                            bool count_include_pad) {
                        return _masked_avg_pool2d(data, mask, kernel_size, stride, padding,
                                                  ceil_mode, count_include_pad);
                    }
                };

                struct masked_avg_pool3d {
                    static inline std::tuple<at::Tensor, at::Tensor> call(
                            const at::Tensor &data,
                            const at::Tensor &mask,
                            at::IntArrayRef kernel_size,
                            at::IntArrayRef stride,
                            at::IntArrayRef padding,
                            bool ceil_mode,
                            bool count_include_pad) {
                        return _masked_avg_pool3d(data, mask, kernel_size, stride, padding,
                                                  ceil_mode, count_include_pad);
                    }
                };

                struct masked_adaptive_avg_pool1d {
                    static inline std::tuple<at::Tensor, at::Tensor> call(
                            const at::Tensor &data,
                            const at::Tensor &mask,
                            at::IntArrayRef output_size) {
                        return _masked_adaptive_avg_pool1d(data, mask, output_size);
                    }
                };

                struct masked_adaptive_avg_pool2d {
                    static inline std::tuple<at::Tensor, at::Tensor> call(
                            const at::Tensor &data,
                            const at::Tensor &mask,
                            at::SymIntArrayRef output_size) {
                        return _masked_adaptive_avg_pool2d(data, mask, at::asIntArrayRefUnchecked(output_size));
                    }
                };

                struct masked_adaptive_avg_pool3d {
                    static inline std::tuple<at::Tensor, at::Tensor> call(
                            const at::Tensor &data,
                            const at::Tensor &mask,
                            at::SymIntArrayRef output_size) {
                        return _masked_adaptive_avg_pool3d(data, mask, at::asIntArrayRefUnchecked(output_size));
                    }
                };
            }

            template<typename aten_op_T, typename pt_op_T, typename input_T, typename... Args>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> avg_poolnd_impl(
                    aten_op_T &&aten_op,
                    pt_op_T &&pt_op,
                    const input_T &input,
                    Args &&... args) {
                if (!utils::has_tensor_mask(input))
                    return masked_pair(aten_op.call(utils::get_data(input), args...));

                at::Tensor output_data, output_mask;
                std::tie(output_data, output_mask) = pt_op.call(utils::get_data(input),
                                                                utils::get_tensor_mask(input), args...);
                return masked_pair(output_data, output_mask);
            }

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

        // avg_pool
        PT_DEFINE_AVG_POOLND_OPS_FORALL_TENSOR_OVERLOADS(
                avg_pool1d, at::_ops::avg_pool1d(), impl::_ops::masked_avg_pool1d())

        PT_DEFINE_AVG_POOLND_OPS_FORALL_TENSOR_OVERLOADS(
                avg_pool2d, impl::_ops::avg_pool2d_no_divisor_override(), impl::_ops::masked_avg_pool2d())

        PT_DEFINE_AVG_POOLND_OPS_FORALL_TENSOR_OVERLOADS(
                avg_pool3d, impl::_ops::avg_pool3d_no_divisor_override(), impl::_ops::masked_avg_pool3d())

        PT_DEFINE_ADAPTIVE_AVG_POOLND_OPS_FORALL_TENSOR_OVERLOADS(
                adaptive_avg_pool1d, at::_ops::adaptive_avg_pool1d(), impl::_ops::masked_adaptive_avg_pool1d(),
                at::IntArrayRef)

        PT_DEFINE_ADAPTIVE_AVG_POOLND_OPS_FORALL_TENSOR_OVERLOADS(
                adaptive_avg_pool2d, at::_ops::adaptive_avg_pool2d(), impl::_ops::masked_adaptive_avg_pool2d(),
                at::SymIntArrayRef)

        PT_DEFINE_ADAPTIVE_AVG_POOLND_OPS_FORALL_TENSOR_OVERLOADS(
                adaptive_avg_pool3d, at::_ops::adaptive_avg_pool3d(), impl::_ops::masked_adaptive_avg_pool3d(),
                at::SymIntArrayRef)

        // max_pool
        // TODO: maybe implement any_poolnd for mask_ops
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

        PT_DEFINE_ADAPTIVE_MAX_POOLND_WITH_INDICES_OPS_FORALL_TENSOR_OVERLOADS(
                adaptive_max_pool1d, at::_ops::adaptive_max_pool1d(), nullptr)

        PT_DEFINE_ADAPTIVE_MAX_POOLND_WITH_INDICES_OPS_FORALL_TENSOR_OVERLOADS(
                adaptive_max_pool2d, at::_ops::adaptive_max_pool2d(), nullptr)

        PT_DEFINE_ADAPTIVE_MAX_POOLND_WITH_INDICES_OPS_FORALL_TENSOR_OVERLOADS(
                adaptive_max_pool3d, at::_ops::adaptive_max_pool3d(), nullptr)

        PT_DEFINE_FRACTIONAL_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(
                fractional_max_pool2d, at::_ops::fractional_max_pool2d(), nullptr)

        PT_DEFINE_FRACTIONAL_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(
                fractional_max_pool3d, at::_ops::fractional_max_pool3d(), nullptr)

        TORCH_LIBRARY_FRAGMENT(partialtorch, m) {
            // avg_pool
            PT_REGISTER_AVG_POOLND_OPS_FORALL_TENSOR_OVERLOADS(avg_pool1d, , 1)
            PT_REGISTER_AVG_POOLND_OPS_FORALL_TENSOR_OVERLOADS(avg_pool2d, , 2)
            PT_REGISTER_AVG_POOLND_OPS_FORALL_TENSOR_OVERLOADS(avg_pool3d, , 3)
            PT_REGISTER_ADAPTIVE_AVG_POOLND_OPS_FORALL_TENSOR_OVERLOADS(adaptive_avg_pool1d, , 1, at::IntArrayRef)
            PT_REGISTER_ADAPTIVE_AVG_POOLND_OPS_FORALL_TENSOR_OVERLOADS(adaptive_avg_pool2d, , 2, at::SymIntArrayRef)
            PT_REGISTER_ADAPTIVE_AVG_POOLND_OPS_FORALL_TENSOR_OVERLOADS(adaptive_avg_pool3d, , 3, at::SymIntArrayRef)
            // max_pool
            PT_REGISTER_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(max_pool1d, , 1)
            PT_REGISTER_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(max_pool2d, , 2)
            PT_REGISTER_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(max_pool3d, , 3)
            PT_REGISTER_MAX_POOLND_WITH_INDICES_OPS_FORALL_TENSOR_OVERLOADS(max_pool1d_with_indices, , 1)
            PT_REGISTER_MAX_POOLND_WITH_INDICES_OPS_FORALL_TENSOR_OVERLOADS(max_pool2d_with_indices, , 2)
            PT_REGISTER_MAX_POOLND_WITH_INDICES_OPS_FORALL_TENSOR_OVERLOADS(max_pool3d_with_indices, , 3)
            PT_REGISTER_ADAPTIVE_MAX_POOLND_WITH_INDICES_OPS_FORALL_TENSOR_OVERLOADS(adaptive_max_pool1d, , 1)
            PT_REGISTER_ADAPTIVE_MAX_POOLND_WITH_INDICES_OPS_FORALL_TENSOR_OVERLOADS(adaptive_max_pool2d, , 2)
            PT_REGISTER_ADAPTIVE_MAX_POOLND_WITH_INDICES_OPS_FORALL_TENSOR_OVERLOADS(adaptive_max_pool3d, , 3)
            PT_REGISTER_FRACTIONAL_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(fractional_max_pool2d, , 2)
            PT_REGISTER_FRACTIONAL_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(fractional_max_pool3d, , 3)
        }
    }
}
