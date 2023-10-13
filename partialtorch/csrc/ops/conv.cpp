#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <torch/library.h>

#include "../MaskedPair.h"
#include "izero_div.h"
#include "reduction.h"
#include "utils/array_ref_utils.h"
#include "utils/fill_identity.h"
#include "utils/irepeat.h"
#include "utils/mask_utils.h"
#include "utils/schema_utils.h"

// ~~~~~ ops definition macros ~~~~~
#define PT_DEFINE_CONVND_OP(NAME, IMPL_OP, INPUT_T, PADDING_T) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                     \
        INPUT_T input,                                         \
        const at::Tensor &weight,                              \
        const c10::optional<at::Tensor> &bias,                 \
        at::IntArrayRef stride,                                \
        PADDING_T padding,                                     \
        at::IntArrayRef dilation,                              \
        int64_t groups) {                                      \
    static constexpr auto op = IMPL_OP;                        \
    return impl::partial_convnd_impl<false>(                   \
            op, input, weight, bias, stride, padding, dilation, groups); \
}

#define PT_DEFINE_SCALED_CONVND_OP(NAME, IMPL_OP, INPUT_T, PADDING_T) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                            \
        INPUT_T input,                                                \
        const at::Tensor &weight,                                     \
        const c10::optional<at::Tensor> &bias,                        \
        at::IntArrayRef stride,                                       \
        PADDING_T padding,                                            \
        at::IntArrayRef dilation,                                     \
        int64_t groups,                                               \
        bool scaled) {                                                \
    static constexpr auto op = IMPL_OP;                               \
    if (scaled)                                                       \
        return impl::partial_convnd_impl<true>(                       \
                op, input, weight, bias, stride, padding, dilation, groups); \
    return impl::partial_convnd_impl<false>(                          \
            op, input, weight, bias, stride, padding, dilation, groups);     \
}

#define PT_DEFINE_CONV_TRANSPOSEND_OP(NAME, IMPL_OP, INPUT_T) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                    \
        INPUT_T input,                                        \
        const at::Tensor &weight,                             \
        const c10::optional<at::Tensor> &bias,                \
        at::IntArrayRef stride,                               \
        c10::SymIntArrayRef padding,                          \
        c10::SymIntArrayRef output_padding,                   \
        int64_t groups,                                       \
        at::IntArrayRef dilation) {                           \
    static constexpr auto op = IMPL_OP;                       \
    return impl::partial_conv_transposend_impl<false>(        \
            op, input, weight, bias, stride, padding, output_padding, groups, dilation); \
}

#define PT_DEFINE_SCALED_CONV_TRANSPOSEND_OP(NAME, IMPL_OP, INPUT_T) \
c10::intrusive_ptr<TensorMaskedPair> NAME(                           \
        INPUT_T input,                                               \
        const at::Tensor &weight,                                    \
        const c10::optional<at::Tensor> &bias,                       \
        at::IntArrayRef stride,                                      \
        c10::SymIntArrayRef padding,                                 \
        c10::SymIntArrayRef output_padding,                          \
        int64_t groups,                                              \
        at::IntArrayRef dilation,                                    \
        bool scaled) {                                               \
    static constexpr auto op = IMPL_OP;                              \
    if (scaled)                                                      \
        return impl::partial_conv_transposend_impl<true>(            \
                op, input, weight, bias, stride, padding, output_padding, groups, dilation); \
    return impl::partial_conv_transposend_impl<false>(               \
            op, input, weight, bias, stride, padding, output_padding, groups, dilation);     \
}

#define PT_DEFINE_CONVND_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, PADDING_T) \
PT_DEFINE_CONVND_OP(NAME, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, PADDING_T) \
PT_DEFINE_CONVND_OP(NAME, IMPL_OP, const at::Tensor &, PADDING_T)

#define PT_DEFINE_SCALED_CONVND_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, PADDING_T) \
PT_DEFINE_SCALED_CONVND_OP(NAME, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>, PADDING_T) \
PT_DEFINE_SCALED_CONVND_OP(NAME, IMPL_OP, const at::Tensor &, PADDING_T)

#define PT_DEFINE_CONVND_OPS_AND_SCALED_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, PADDING_T) \
PT_DEFINE_CONVND_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, PADDING_T)                        \
PT_DEFINE_SCALED_CONVND_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP, PADDING_T)

#define PT_DEFINE_CONV_TRANSPOSEND_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP) \
PT_DEFINE_CONV_TRANSPOSEND_OP(NAME, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DEFINE_CONV_TRANSPOSEND_OP(NAME, IMPL_OP, const at::Tensor &)

#define PT_DEFINE_SCALED_CONV_TRANSPOSEND_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP) \
PT_DEFINE_SCALED_CONV_TRANSPOSEND_OP(NAME, IMPL_OP, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DEFINE_SCALED_CONV_TRANSPOSEND_OP(NAME, IMPL_OP, const at::Tensor &)

#define PT_DEFINE_CONV_TRANSPOSEND_OPS_AND_SCALED_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP) \
PT_DEFINE_CONV_TRANSPOSEND_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP)                        \
PT_DEFINE_SCALED_CONV_TRANSPOSEND_OPS_FORALL_TENSOR_OVERLOADS(NAME, IMPL_OP)

// ~~~~~ ops registration macros ~~~~~
#define PT_CONVND_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T, PADDING_T, PADDING_DEFAULT) \
utils::FunctionSchemaBuilder(#NAME).add_overload(#OVERLOAD_NAME).add_overload(#POSTFIX).arg<INPUT_T>("input").arg<const at::Tensor &>("weight").arg<const c10::optional<at::Tensor> &>("bias", "None").arg<at::IntArrayRef, DIMENSION>("stride", "1").arg<PADDING_T, DIMENSION>("padding", #PADDING_DEFAULT).arg<at::IntArrayRef, DIMENSION>("dilation", "1").arg<int64_t>("groups", "1").ret<TensorMaskedPair>()

#define PT_CONV_TRANSPOSEND_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T) \
utils::FunctionSchemaBuilder(#NAME).add_overload(#OVERLOAD_NAME).add_overload(#POSTFIX).arg<INPUT_T>("input").arg<const at::Tensor &>("weight").arg<const c10::optional<at::Tensor> &>("bias", "None").arg<at::IntArrayRef, DIMENSION>("stride", "1").arg<at::SymIntArrayRef, DIMENSION>("padding", "0").arg<at::SymIntArrayRef, DIMENSION>("output_padding", "0").arg<int64_t>("groups", "1").arg<at::IntArrayRef, DIMENSION>("dilation", "1").ret<TensorMaskedPair>()

#define PT_REGISTER_CONVND_OP(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T, PADDING_T, PADDING_DEFAULT) \
m.def(PT_CONVND_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T, PADDING_T, PADDING_DEFAULT).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(INPUT_T, const at::Tensor &, const c10::optional<at::Tensor> &, at::IntArrayRef, PADDING_T, at::IntArrayRef, int64_t)>(NAME)));

#define PT_REGISTER_SCALED_CONVND_OP(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T, PADDING_T, PADDING_DEFAULT) \
m.def(PT_CONVND_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T, PADDING_T, PADDING_DEFAULT).add_overload("scaled").vararg().arg<bool>("scaled").schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(INPUT_T, const at::Tensor &, const c10::optional<at::Tensor> &, at::IntArrayRef, PADDING_T, at::IntArrayRef, int64_t, bool)>(NAME)));

#define PT_REGISTER_CONV_TRANSPOSEND_OP(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T) \
m.def(PT_CONV_TRANSPOSEND_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T).schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(INPUT_T, const at::Tensor &, const c10::optional<at::Tensor> &, at::IntArrayRef, c10::SymIntArrayRef, c10::SymIntArrayRef, int64_t, at::IntArrayRef)>(NAME)));

#define PT_REGISTER_SCALED_CONV_TRANSPOSEND_OP(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T) \
m.def(PT_CONV_TRANSPOSEND_SCHEMA_BUILDER(NAME, OVERLOAD_NAME, POSTFIX, DIMENSION, INPUT_T).add_overload("scaled").vararg().arg<bool>("scaled").schema().c_str(), \
    TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(INPUT_T, const at::Tensor &, const c10::optional<at::Tensor> &, at::IntArrayRef, c10::SymIntArrayRef, c10::SymIntArrayRef, int64_t, at::IntArrayRef, bool)>(NAME)));

#define PT_REGISTER_CONVND_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, DIMENSION, PADDING_T, PADDING_DEFAULT) \
PT_REGISTER_CONVND_OP(NAME, MaskedPair, POSTFIX, DIMENSION, const_intrusive_ptr_arg_t<TensorMaskedPair>, PADDING_T, PADDING_DEFAULT) \
PT_REGISTER_CONVND_OP(NAME, Tensor, POSTFIX, DIMENSION, const at::Tensor &, PADDING_T, PADDING_DEFAULT)

#define PT_REGISTER_SCALED_CONVND_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, DIMENSION, PADDING_T, PADDING_DEFAULT) \
PT_REGISTER_SCALED_CONVND_OP(NAME, MaskedPair, POSTFIX, DIMENSION, const_intrusive_ptr_arg_t<TensorMaskedPair>, PADDING_T, PADDING_DEFAULT) \
PT_REGISTER_SCALED_CONVND_OP(NAME, Tensor, POSTFIX, DIMENSION, const at::Tensor &, PADDING_T, PADDING_DEFAULT)

#define PT_REGISTER_CONVND_OPS_AND_SCALED_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, DIMENSION, PADDING_T, PADDING_DEFAULT) \
PT_REGISTER_CONVND_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, DIMENSION, PADDING_T, PADDING_DEFAULT)                        \
PT_REGISTER_SCALED_CONVND_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, DIMENSION, PADDING_T, PADDING_DEFAULT)

#define PT_REGISTER_CONV_TRANSPOSEND_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, DIMENSION) \
PT_REGISTER_CONV_TRANSPOSEND_OP(NAME, MaskedPair, POSTFIX, DIMENSION, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_REGISTER_CONV_TRANSPOSEND_OP(NAME, Tensor, POSTFIX, DIMENSION, const at::Tensor &)

#define PT_REGISTER_SCALED_CONV_TRANSPOSEND_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, DIMENSION) \
PT_REGISTER_SCALED_CONV_TRANSPOSEND_OP(NAME, MaskedPair, POSTFIX, DIMENSION, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_REGISTER_SCALED_CONV_TRANSPOSEND_OP(NAME, Tensor, POSTFIX, DIMENSION, const at::Tensor &)

#define PT_REGISTER_CONV_TRANSPOSEND_OPS_AND_SCALED_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, DIMENSION) \
PT_REGISTER_CONV_TRANSPOSEND_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, DIMENSION)                        \
PT_REGISTER_SCALED_CONV_TRANSPOSEND_OPS_FORALL_TENSOR_OVERLOADS(NAME, POSTFIX, DIMENSION)

namespace partialtorch {
    namespace ops {
        namespace impl {
            struct convnd_canonical_scale {
                static int64_t call(const at::Tensor &weight, int64_t groups) {
                    int64_t res = weight.size(1);
                    for (const auto d: c10::irange(2, weight.ndimension()))
                        res *= weight.size(d);
                    return res / groups;
                }
            };

            struct conv_transposend_canonical_scale {
                static int64_t call(const at::Tensor &weight, int64_t groups) {
                    int64_t res = weight.size(1);
                    for (const auto d: c10::irange(2, weight.ndimension()))
                        res *= weight.size(d);
                    return res * groups;
                }
            };

            struct convolution_canonical_scale {
                static int64_t call(const at::Tensor &weight, bool transposed, int64_t groups) {
                    return transposed ? conv_transposend_canonical_scale::call(weight, groups)
                                      : convnd_canonical_scale::call(weight, groups);
                }
            };

            template<bool scaled = false, typename input_T, typename padding_T>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> partial_convolution_impl(
                    const input_T &input,
                    const at::Tensor &weight,
                    const c10::optional<at::Tensor> &bias,
                    at::IntArrayRef stride,
                    padding_T padding,
                    at::IntArrayRef dilation,
                    bool transposed,
                    padding_T output_padding,
                    int64_t groups) {
                if (!utils::has_tensor_mask(input))
                    return masked_pair(at::_ops::convolution::call(
                            utils::get_data(input), weight, bias,
                            stride, utils::to_sym_int(padding), dilation, transposed,
                            utils::to_sym_int(output_padding), groups));

                bool unbatched = utils::get_data(input).ndimension() == weight.ndimension() - 1;
                auto output_data = at::_ops::convolution::call(
                        utils::_ops::fill_identity_zeros<false>::call(input), weight, {},
                        stride, utils::to_sym_int(padding), dilation, transposed,
                        utils::to_sym_int(output_padding), groups);
                auto mask_ratio_options = output_data.options();
                at::Tensor mask_ratio;
                {
                    at::NoGradGuard g;
                    mask_ratio = at::_ops::convolution::call(
                            utils::get_tensor_mask(input, mask_ratio_options), at::ones_like(weight), {},
                            stride, utils::to_sym_int(padding), dilation, transposed,
                            utils::to_sym_int(output_padding), groups);
                    if constexpr (scaled)
                        utils::izero_ldiv_(
                                mask_ratio,
                                convolution_canonical_scale::call(weight, transposed, groups));
                }
                if constexpr (scaled)
                    output_data.mul_(mask_ratio);
                if (bias.has_value()) {
                    std::vector<int64_t> bias_view_shape;
                    if (!unbatched) {
                        bias_view_shape.reserve(weight.ndimension());
                        bias_view_shape.emplace_back(1);
                    } else
                        bias_view_shape.reserve(weight.ndimension() - 1);
                    bias_view_shape.emplace_back(-1);
                    for (const auto i: c10::irepeat(1, weight.ndimension() - 2))
                        bias_view_shape.emplace_back(i);

                    output_data.add_(bias->view(bias_view_shape));
                }
                auto output_mask = mask_ratio.to(at::kBool);
                return masked_pair(output_data, output_mask);
            }

            template<bool scaled = false, typename op_T, typename input_T, typename stride_T, typename padding_T>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> partial_convnd_impl(
                    op_T &&op,
                    const input_T &input,
                    const at::Tensor &weight,
                    const c10::optional<at::Tensor> &bias,
                    stride_T stride,
                    padding_T padding,
                    at::IntArrayRef dilation,
                    int64_t groups) {
                if (!utils::has_tensor_mask(input))
                    return masked_pair(op.call(
                            utils::get_data(input), weight, bias,
                            stride, padding, dilation, groups));

                bool unbatched = utils::get_data(input).ndimension() == weight.ndimension() - 1;
                auto output_data = op.call(
                        utils::_ops::fill_identity_zeros<false>::call(input), weight, {},
                        stride, padding, dilation, groups);
                auto mask_ratio_options = output_data.options();
                at::Tensor mask_ratio;
                {
                    at::NoGradGuard g;
                    mask_ratio = op.call(
                            utils::get_tensor_mask(input, mask_ratio_options), at::ones_like(weight), {},
                            stride, padding, dilation, groups);
                    if constexpr (scaled)
                        utils::izero_ldiv_(
                                mask_ratio,
                                convnd_canonical_scale::call(weight, groups));
                }
                if constexpr (scaled)
                    output_data.mul_(mask_ratio);
                if (bias.has_value()) {
                    std::vector<int64_t> bias_view_shape;
                    if (!unbatched) {
                        bias_view_shape.reserve(weight.ndimension());
                        bias_view_shape.emplace_back(1);
                    } else
                        bias_view_shape.reserve(weight.ndimension() - 1);
                    bias_view_shape.emplace_back(-1);
                    for (const auto i: c10::irepeat(1, weight.ndimension() - 2))
                        bias_view_shape.emplace_back(i);

                    output_data.add_(bias->view(bias_view_shape));
                }
                auto output_mask = mask_ratio.to(at::kBool);
                return masked_pair(output_data, output_mask);
            }

            template<bool scaled = false, typename op_T, typename input_T,
                    typename stride_T, typename padding_T, typename output_padding_T>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> partial_conv_transposend_impl(
                    op_T &&op,
                    const input_T &input,
                    const at::Tensor &weight,
                    const c10::optional<at::Tensor> &bias,
                    stride_T stride,
                    padding_T padding,
                    output_padding_T output_padding,
                    int64_t groups,
                    at::IntArrayRef dilation) {
                if (!utils::has_tensor_mask(input))
                    return masked_pair(op.call(
                            utils::get_data(input), weight, bias,
                            stride, padding, output_padding, groups, dilation));

                bool unbatched = utils::get_data(input).ndimension() == weight.ndimension() - 1;
                auto output_data = op.call(
                        utils::_ops::fill_identity_zeros<false>::call(input), weight, {},
                        stride, padding, output_padding, groups, dilation);
                auto mask_ratio_options = output_data.options();
                at::Tensor mask_ratio;
                {
                    at::NoGradGuard g;
                    mask_ratio = op.call(
                            utils::get_tensor_mask(input, mask_ratio_options), at::ones_like(weight), {},
                            stride, padding, output_padding, groups, dilation);
                    if constexpr (scaled)
                        utils::izero_ldiv_(
                                mask_ratio,
                                conv_transposend_canonical_scale::call(weight, groups));
                }
                if constexpr (scaled)
                    output_data.mul_(mask_ratio);
                if (bias.has_value()) {
                    std::vector<int64_t> bias_view_shape;
                    if (!unbatched) {
                        bias_view_shape.reserve(weight.ndimension());
                        bias_view_shape.emplace_back(1);
                    } else
                        bias_view_shape.reserve(weight.ndimension() - 1);
                    bias_view_shape.emplace_back(-1);
                    for (const auto i: c10::irepeat(1, weight.ndimension() - 2))
                        bias_view_shape.emplace_back(i);

                    output_data.add_(bias->view(bias_view_shape));
                }
                auto output_mask = mask_ratio.to(at::kBool);
                return masked_pair(output_data, output_mask);
            }
        }

        // convolution
        c10::intrusive_ptr<TensorMaskedPair> partial_convolution(
                const_intrusive_ptr_arg_t<TensorMaskedPair> input,
                const at::Tensor &weight,
                const c10::optional<at::Tensor> &bias,
                at::IntArrayRef stride,
                at::IntArrayRef padding,
                at::IntArrayRef dilation,
                bool transposed,
                at::IntArrayRef output_padding,
                int64_t groups) {
            return impl::partial_convolution_impl<false>(
                    input, weight, bias, stride, padding, dilation,
                    transposed, output_padding, groups);
        }

        c10::intrusive_ptr<TensorMaskedPair> partial_convolution(
                const at::Tensor &input,
                const at::Tensor &weight,
                const c10::optional<at::Tensor> &bias,
                at::IntArrayRef stride,
                at::IntArrayRef padding,
                at::IntArrayRef dilation,
                bool transposed,
                at::IntArrayRef output_padding,
                int64_t groups) {
            return impl::partial_convolution_impl<false>(
                    input, weight, bias, stride, padding, dilation,
                    transposed, output_padding, groups);
        }

        c10::intrusive_ptr<TensorMaskedPair> partial_convolution(
                const_intrusive_ptr_arg_t<TensorMaskedPair> input,
                const at::Tensor &weight,
                const c10::optional<at::Tensor> &bias,
                at::IntArrayRef stride,
                c10::SymIntArrayRef padding,
                at::IntArrayRef dilation,
                bool transposed,
                c10::SymIntArrayRef output_padding,
                int64_t groups) {
            return impl::partial_convolution_impl<false>(
                    input, weight, bias, stride, padding, dilation,
                    transposed, output_padding, groups);
        }

        c10::intrusive_ptr<TensorMaskedPair> partial_convolution(
                const at::Tensor &input,
                const at::Tensor &weight,
                const c10::optional<at::Tensor> &bias,
                at::IntArrayRef stride,
                c10::SymIntArrayRef padding,
                at::IntArrayRef dilation,
                bool transposed,
                c10::SymIntArrayRef output_padding,
                int64_t groups) {
            return impl::partial_convolution_impl<false>(
                    input, weight, bias, stride, padding, dilation,
                    transposed, output_padding, groups);
        }

        // scaled convolution
        c10::intrusive_ptr<TensorMaskedPair> partial_convolution(
                const_intrusive_ptr_arg_t<TensorMaskedPair> input,
                const at::Tensor &weight,
                const c10::optional<at::Tensor> &bias,
                at::IntArrayRef stride,
                at::IntArrayRef padding,
                at::IntArrayRef dilation,
                bool transposed,
                at::IntArrayRef output_padding,
                int64_t groups,
                bool scaled) {
            if (scaled)
                return impl::partial_convolution_impl<true>(
                        input, weight, bias, stride, padding, dilation,
                        transposed, output_padding, groups);
            return impl::partial_convolution_impl<false>(
                    input, weight, bias, stride, padding, dilation,
                    transposed, output_padding, groups);
        }

        c10::intrusive_ptr<TensorMaskedPair> partial_convolution(
                const at::Tensor &input,
                const at::Tensor &weight,
                const c10::optional<at::Tensor> &bias,
                at::IntArrayRef stride,
                at::IntArrayRef padding,
                at::IntArrayRef dilation,
                bool transposed,
                at::IntArrayRef output_padding,
                int64_t groups,
                bool scaled) {
            if (scaled)
                return impl::partial_convolution_impl<true>(
                        input, weight, bias, stride, padding, dilation,
                        transposed, output_padding, groups);
            return impl::partial_convolution_impl<false>(
                    input, weight, bias, stride, padding, dilation,
                    transposed, output_padding, groups);
        }

        c10::intrusive_ptr<TensorMaskedPair> partial_convolution(
                const_intrusive_ptr_arg_t<TensorMaskedPair> input,
                const at::Tensor &weight,
                const c10::optional<at::Tensor> &bias,
                at::IntArrayRef stride,
                c10::SymIntArrayRef padding,
                at::IntArrayRef dilation,
                bool transposed,
                c10::SymIntArrayRef output_padding,
                int64_t groups,
                bool scaled) {
            if (scaled)
                return impl::partial_convolution_impl<true>(
                        input, weight, bias, stride, padding, dilation,
                        transposed, output_padding, groups);
            return impl::partial_convolution_impl<false>(
                    input, weight, bias, stride, padding, dilation,
                    transposed, output_padding, groups);
        }

        c10::intrusive_ptr<TensorMaskedPair> partial_convolution(
                const at::Tensor &input,
                const at::Tensor &weight,
                const c10::optional<at::Tensor> &bias,
                at::IntArrayRef stride,
                c10::SymIntArrayRef padding,
                at::IntArrayRef dilation,
                bool transposed,
                c10::SymIntArrayRef output_padding,
                int64_t groups,
                bool scaled) {
            if (scaled)
                return impl::partial_convolution_impl<true>(
                        input, weight, bias, stride, padding, dilation,
                        transposed, output_padding, groups);
            return impl::partial_convolution_impl<false>(
                    input, weight, bias, stride, padding, dilation,
                    transposed, output_padding, groups);
        }

        // convnd
        PT_DEFINE_CONVND_OPS_AND_SCALED_OPS_FORALL_TENSOR_OVERLOADS(
                partial_conv1d, at::_ops::conv1d(), c10::SymIntArrayRef)

        PT_DEFINE_CONVND_OPS_AND_SCALED_OPS_FORALL_TENSOR_OVERLOADS(
                partial_conv1d, at::_ops::conv1d_padding(), c10::string_view)

        PT_DEFINE_CONVND_OPS_AND_SCALED_OPS_FORALL_TENSOR_OVERLOADS(
                partial_conv2d, at::_ops::conv2d(), c10::SymIntArrayRef)

        PT_DEFINE_CONVND_OPS_AND_SCALED_OPS_FORALL_TENSOR_OVERLOADS(
                partial_conv2d, at::_ops::conv2d_padding(), c10::string_view)

        PT_DEFINE_CONVND_OPS_AND_SCALED_OPS_FORALL_TENSOR_OVERLOADS(
                partial_conv3d, at::_ops::conv3d(), c10::SymIntArrayRef)

        PT_DEFINE_CONVND_OPS_AND_SCALED_OPS_FORALL_TENSOR_OVERLOADS(
                partial_conv3d, at::_ops::conv3d_padding(), c10::string_view)

        // conv_transposend
        PT_DEFINE_CONV_TRANSPOSEND_OPS_AND_SCALED_OPS_FORALL_TENSOR_OVERLOADS(
                partial_conv_transpose1d, at::_ops::conv_transpose1d())

        PT_DEFINE_CONV_TRANSPOSEND_OPS_AND_SCALED_OPS_FORALL_TENSOR_OVERLOADS(
                partial_conv_transpose2d, at::_ops::conv_transpose2d_input())

        PT_DEFINE_CONV_TRANSPOSEND_OPS_AND_SCALED_OPS_FORALL_TENSOR_OVERLOADS(
                partial_conv_transpose3d, at::_ops::conv_transpose3d_input())

        TORCH_LIBRARY_FRAGMENT(partialtorch, m) {
            // convolution
            m.def(utils::FunctionSchemaBuilder("partial_convolution").overload("MaskedPair")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("input")
                          .arg<const at::Tensor &>("weight")
                          .arg<const c10::optional<at::Tensor> &>("bias")
                          .arg<at::IntArrayRef>("stride")
                          .arg<at::IntArrayRef>("padding")
                          .arg<at::IntArrayRef>("dilation")
                          .arg<bool>("transposed")
                          .arg<at::IntArrayRef>("output_padding")
                          .arg<int64_t>("groups")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          const at::Tensor &,
                          const c10::optional<at::Tensor> &,
                          at::IntArrayRef,
                          at::IntArrayRef,
                          at::IntArrayRef,
                          bool,
                          at::IntArrayRef,
                          int64_t)>(partial_convolution)));
            m.def(utils::FunctionSchemaBuilder("partial_convolution").overload("Tensor")
                          .arg<const at::Tensor &>("input")
                          .arg<const at::Tensor &>("weight")
                          .arg<const c10::optional<at::Tensor> &>("bias")
                          .arg<at::IntArrayRef>("stride")
                          .arg<at::IntArrayRef>("padding")
                          .arg<at::IntArrayRef>("dilation")
                          .arg<bool>("transposed")
                          .arg<at::IntArrayRef>("output_padding")
                          .arg<int64_t>("groups")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          const at::Tensor &,
                          const c10::optional<at::Tensor> &,
                          at::IntArrayRef,
                          at::IntArrayRef,
                          at::IntArrayRef,
                          bool,
                          at::IntArrayRef,
                          int64_t)>(partial_convolution)));
            m.def(utils::FunctionSchemaBuilder("partial_convolution").overload("MaskedPair_scaled")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("input")
                          .arg<const at::Tensor &>("weight")
                          .arg<const c10::optional<at::Tensor> &>("bias")
                          .arg<at::IntArrayRef>("stride")
                          .arg<at::IntArrayRef>("padding")
                          .arg<at::IntArrayRef>("dilation")
                          .arg<bool>("transposed")
                          .arg<at::IntArrayRef>("output_padding")
                          .arg<int64_t>("groups")
                          .vararg().arg<bool>("scaled")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          const at::Tensor &,
                          const c10::optional<at::Tensor> &,
                          at::IntArrayRef,
                          at::IntArrayRef,
                          at::IntArrayRef,
                          bool,
                          at::IntArrayRef,
                          int64_t,
                          bool)>(partial_convolution)));
            m.def(utils::FunctionSchemaBuilder("partial_convolution").overload("Tensor_scaled")
                          .arg<const at::Tensor &>("input")
                          .arg<const at::Tensor &>("weight")
                          .arg<const c10::optional<at::Tensor> &>("bias")
                          .arg<at::IntArrayRef>("stride")
                          .arg<at::IntArrayRef>("padding")
                          .arg<at::IntArrayRef>("dilation")
                          .arg<bool>("transposed")
                          .arg<at::IntArrayRef>("output_padding")
                          .arg<int64_t>("groups")
                          .vararg().arg<bool>("scaled")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          const at::Tensor &,
                          const c10::optional<at::Tensor> &,
                          at::IntArrayRef,
                          at::IntArrayRef,
                          at::IntArrayRef,
                          bool,
                          at::IntArrayRef,
                          int64_t,
                          bool)>(partial_convolution)));

            m.def(utils::FunctionSchemaBuilder("partial_convolution").overload("MaskedPair_symint")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("input")
                          .arg<const at::Tensor &>("weight")
                          .arg<const c10::optional<at::Tensor> &>("bias")
                          .arg<at::IntArrayRef>("stride")
                          .arg<c10::SymIntArrayRef>("padding")
                          .arg<at::IntArrayRef>("dilation")
                          .arg<bool>("transposed")
                          .arg<c10::SymIntArrayRef>("output_padding")
                          .arg<int64_t>("groups")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          const at::Tensor &,
                          const c10::optional<at::Tensor> &,
                          at::IntArrayRef,
                          c10::SymIntArrayRef,
                          at::IntArrayRef,
                          bool,
                          c10::SymIntArrayRef,
                          int64_t)>(partial_convolution)));
            m.def(utils::FunctionSchemaBuilder("partial_convolution").overload("Tensor_symint")
                          .arg<const at::Tensor &>("input")
                          .arg<const at::Tensor &>("weight")
                          .arg<const c10::optional<at::Tensor> &>("bias")
                          .arg<at::IntArrayRef>("stride")
                          .arg<c10::SymIntArrayRef>("padding")
                          .arg<at::IntArrayRef>("dilation")
                          .arg<bool>("transposed")
                          .arg<c10::SymIntArrayRef>("output_padding")
                          .arg<int64_t>("groups")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          const at::Tensor &,
                          const c10::optional<at::Tensor> &,
                          at::IntArrayRef,
                          c10::SymIntArrayRef,
                          at::IntArrayRef,
                          bool,
                          c10::SymIntArrayRef,
                          int64_t)>(partial_convolution)));
            m.def(utils::FunctionSchemaBuilder("partial_convolution").overload("MaskedPair_symint_scaled")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("input")
                          .arg<const at::Tensor &>("weight")
                          .arg<const c10::optional<at::Tensor> &>("bias")
                          .arg<at::IntArrayRef>("stride")
                          .arg<c10::SymIntArrayRef>("padding")
                          .arg<at::IntArrayRef>("dilation")
                          .arg<bool>("transposed")
                          .arg<c10::SymIntArrayRef>("output_padding")
                          .arg<int64_t>("groups")
                          .vararg().arg<bool>("scaled")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          const at::Tensor &,
                          const c10::optional<at::Tensor> &,
                          at::IntArrayRef,
                          c10::SymIntArrayRef,
                          at::IntArrayRef,
                          bool,
                          c10::SymIntArrayRef,
                          int64_t,
                          bool)>(partial_convolution)));
            m.def(utils::FunctionSchemaBuilder("partial_convolution").overload("Tensor_symint_scaled")
                          .arg<const at::Tensor &>("input")
                          .arg<const at::Tensor &>("weight")
                          .arg<const c10::optional<at::Tensor> &>("bias")
                          .arg<at::IntArrayRef>("stride")
                          .arg<c10::SymIntArrayRef>("padding")
                          .arg<at::IntArrayRef>("dilation")
                          .arg<bool>("transposed")
                          .arg<c10::SymIntArrayRef>("output_padding")
                          .arg<int64_t>("groups")
                          .vararg().arg<bool>("scaled")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          const at::Tensor &,
                          const c10::optional<at::Tensor> &,
                          at::IntArrayRef,
                          c10::SymIntArrayRef,
                          at::IntArrayRef,
                          bool,
                          c10::SymIntArrayRef,
                          int64_t,
                          bool)>(partial_convolution)));

            // convnd
            PT_REGISTER_CONVND_OPS_AND_SCALED_OPS_FORALL_TENSOR_OVERLOADS(
                    partial_conv1d, , 1,
                    c10::SymIntArrayRef, 0)
            PT_REGISTER_CONVND_OPS_AND_SCALED_OPS_FORALL_TENSOR_OVERLOADS(
                    partial_conv1d, padding, 1,
                    c10::string_view, "valid")

            PT_REGISTER_CONVND_OPS_AND_SCALED_OPS_FORALL_TENSOR_OVERLOADS(
                    partial_conv2d, , 2,
                    c10::SymIntArrayRef, 0)
            PT_REGISTER_CONVND_OPS_AND_SCALED_OPS_FORALL_TENSOR_OVERLOADS(
                    partial_conv2d, padding, 2,
                    c10::string_view, "valid")

            PT_REGISTER_CONVND_OPS_AND_SCALED_OPS_FORALL_TENSOR_OVERLOADS(
                    partial_conv3d, , 3,
                    c10::SymIntArrayRef, 0)
            PT_REGISTER_CONVND_OPS_AND_SCALED_OPS_FORALL_TENSOR_OVERLOADS(
                    partial_conv3d, padding, 3,
                    c10::string_view, "valid")

            // conv_transposend
            PT_REGISTER_CONV_TRANSPOSEND_OPS_AND_SCALED_OPS_FORALL_TENSOR_OVERLOADS(
                    partial_conv_transpose1d, , 1)
            PT_REGISTER_CONV_TRANSPOSEND_OPS_AND_SCALED_OPS_FORALL_TENSOR_OVERLOADS(
                    partial_conv_transpose2d, , 2)
            PT_REGISTER_CONV_TRANSPOSEND_OPS_AND_SCALED_OPS_FORALL_TENSOR_OVERLOADS(
                    partial_conv_transpose3d, , 3)
        }
    }
}
