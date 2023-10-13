#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <torch/library.h>

#include "../MaskedPair.h"
#include "izero_div.h"
#include "reduction.h"
#include "utils/fill_identity.h"
#include "utils/mask_utils.h"
#include "utils/schema_utils.h"

namespace partialtorch {
    namespace ops {
        namespace impl {
            template<bool scaled = false, typename input_T>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> partial_linear_impl(
                    const input_T &input,
                    const at::Tensor &weight,
                    const c10::optional<at::Tensor> &bias) {
                if (!utils::has_tensor_mask(input))
                    return masked_pair(at::linear(utils::get_data(input), weight, bias));

                if constexpr (scaled) {
                    auto output_data = at::linear(
                            utils::_ops::fill_identity_zeros<false>::call(input), weight);
                    auto mask_ratio_options = output_data.options();
                    at::Tensor mask_ratio;
                    {
                        at::NoGradGuard g;
                        mask_ratio = at::sum(
                                utils::get_tensor_mask(input, mask_ratio_options), -1, true).expand_as(output_data);
                        mask_ratio = utils::izero_ldiv(mask_ratio, weight.size(-1));
                    }
                    output_data.mul_(mask_ratio);
                    if (bias.has_value())
                        output_data.add_(bias.value());
                    return masked_pair(output_data);
                } else {
                    return masked_pair(at::linear(
                            utils::_ops::fill_identity_zeros<false>::call(input), weight, bias));
                }
            }

            template<bool scaled = false, typename input1_T, typename input2_T>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> partial_bilinear_impl(
                    const input1_T &input1,
                    const input2_T &input2,
                    const at::Tensor &weight,
                    const c10::optional<at::Tensor> &bias) {
                if (!utils::has_tensor_mask(input1) && !utils::has_tensor_mask(input2))
                    return masked_pair(at::_ops::bilinear::call(
                            utils::get_data(input1),
                            utils::get_data(input2),
                            weight, bias));

                if constexpr (scaled) {
                    auto output_data = at::bilinear(
                            utils::_ops::fill_identity_zeros<false>::call(input1),
                            utils::_ops::fill_identity_zeros<false>::call(input2),
                            weight);
                    auto mask_ratio_options = output_data.options();
                    at::Tensor mask_ratio;
                    {
                        at::NoGradGuard g;
                        mask_ratio = at::_ops::bilinear::call(
                                utils::get_tensor_mask(input1, mask_ratio_options),
                                utils::get_tensor_mask(input2, mask_ratio_options),
                                at::ones_like(weight), {});
                        utils::izero_ldiv_(mask_ratio, weight.size(-1) * weight.size(-2));
                    }
                    output_data.mul_(mask_ratio);
                    if (bias.has_value())
                        output_data.add_(bias.value());
                    return masked_pair(output_data);
                } else {
                    return masked_pair(at::_ops::bilinear::call(
                            utils::_ops::fill_identity_zeros<false>::call(input1),
                            utils::_ops::fill_identity_zeros<false>::call(input2),
                            weight, bias));
                }
            }
        }

        // linear
        c10::intrusive_ptr<TensorMaskedPair> partial_linear(
                const_intrusive_ptr_arg_t<TensorMaskedPair> input,
                const at::Tensor &weight,
                const c10::optional<at::Tensor> &bias) {
            return impl::partial_linear_impl<false>(input, weight, bias);
        }

        c10::intrusive_ptr<TensorMaskedPair> partial_linear(
                const at::Tensor &input,
                const at::Tensor &weight,
                const c10::optional<at::Tensor> &bias) {
            return impl::partial_linear_impl<false>(input, weight, bias);
        }

        c10::intrusive_ptr<TensorMaskedPair> partial_linear(
                const_intrusive_ptr_arg_t<TensorMaskedPair> input,
                const at::Tensor &weight,
                const c10::optional<at::Tensor> &bias,
                bool scaled) {
            if (scaled)
                return impl::partial_linear_impl<true>(input, weight, bias);
            return impl::partial_linear_impl<false>(input, weight, bias);
        }

        c10::intrusive_ptr<TensorMaskedPair> partial_linear(
                const at::Tensor &input,
                const at::Tensor &weight,
                const c10::optional<at::Tensor> &bias,
                bool scaled) {
            if (scaled)
                return impl::partial_linear_impl<true>(input, weight, bias);
            return impl::partial_linear_impl<false>(input, weight, bias);
        }

        // bilinear
        c10::intrusive_ptr<TensorMaskedPair> partial_bilinear(
                const_intrusive_ptr_arg_t<TensorMaskedPair> input1,
                const_intrusive_ptr_arg_t<TensorMaskedPair> input2,
                const at::Tensor &weight,
                const c10::optional<at::Tensor> &bias) {
            return impl::partial_bilinear_impl<false>(input1, input2, weight, bias);
        }

        c10::intrusive_ptr<TensorMaskedPair> partial_bilinear(
                const_intrusive_ptr_arg_t<TensorMaskedPair> input1,
                const at::Tensor &input2,
                const at::Tensor &weight,
                const c10::optional<at::Tensor> &bias) {
            return impl::partial_bilinear_impl<false>(input1, input2, weight, bias);
        }

        c10::intrusive_ptr<TensorMaskedPair> partial_bilinear(
                const at::Tensor &input1,
                const_intrusive_ptr_arg_t<TensorMaskedPair> input2,
                const at::Tensor &weight,
                const c10::optional<at::Tensor> &bias) {
            return impl::partial_bilinear_impl<false>(input1, input2, weight, bias);
        }

        c10::intrusive_ptr<TensorMaskedPair> partial_bilinear(
                const at::Tensor &input1,
                const at::Tensor &input2,
                const at::Tensor &weight,
                const c10::optional<at::Tensor> &bias) {
            return impl::partial_bilinear_impl<false>(input1, input2, weight, bias);
        }

        c10::intrusive_ptr<TensorMaskedPair> partial_bilinear(
                const_intrusive_ptr_arg_t<TensorMaskedPair> input1,
                const_intrusive_ptr_arg_t<TensorMaskedPair> input2,
                const at::Tensor &weight,
                const c10::optional<at::Tensor> &bias,
                bool scaled) {
            if (scaled)
                return impl::partial_bilinear_impl<true>(input1, input2, weight, bias);
            return impl::partial_bilinear_impl<false>(input1, input2, weight, bias);
        }

        c10::intrusive_ptr<TensorMaskedPair> partial_bilinear(
                const_intrusive_ptr_arg_t<TensorMaskedPair> input1,
                const at::Tensor &input2,
                const at::Tensor &weight,
                const c10::optional<at::Tensor> &bias,
                bool scaled) {
            if (scaled)
                return impl::partial_bilinear_impl<true>(input1, input2, weight, bias);
            return impl::partial_bilinear_impl<false>(input1, input2, weight, bias);
        }

        c10::intrusive_ptr<TensorMaskedPair> partial_bilinear(
                const at::Tensor &input1,
                const_intrusive_ptr_arg_t<TensorMaskedPair> input2,
                const at::Tensor &weight,
                const c10::optional<at::Tensor> &bias,
                bool scaled) {
            if (scaled)
                return impl::partial_bilinear_impl<true>(input1, input2, weight, bias);
            return impl::partial_bilinear_impl<false>(input1, input2, weight, bias);
        }

        c10::intrusive_ptr<TensorMaskedPair> partial_bilinear(
                const at::Tensor &input1,
                const at::Tensor &input2,
                const at::Tensor &weight,
                const c10::optional<at::Tensor> &bias,
                bool scaled) {
            if (scaled)
                return impl::partial_bilinear_impl<true>(input1, input2, weight, bias);
            return impl::partial_bilinear_impl<false>(input1, input2, weight, bias);
        }

        TORCH_LIBRARY_FRAGMENT(partialtorch, m) {
            // linear
            m.def(utils::FunctionSchemaBuilder("partial_linear").overload("MaskedPair")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("input")
                          .arg<const at::Tensor &>("weight")
                          .arg<const c10::optional<at::Tensor> &>("bias", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          const at::Tensor &,
                          const c10::optional<at::Tensor> &)>(partial_linear)));
            m.def(utils::FunctionSchemaBuilder("partial_linear").overload("Tensor")
                          .arg<const at::Tensor &>("input")
                          .arg<const at::Tensor &>("weight")
                          .arg<const c10::optional<at::Tensor> &>("bias", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          const at::Tensor &,
                          const c10::optional<at::Tensor> &)>(partial_linear)));

            m.def(utils::FunctionSchemaBuilder("partial_linear").overload("MaskedPair_scaled")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("input")
                          .arg<const at::Tensor &>("weight")
                          .arg<const c10::optional<at::Tensor> &>("bias", "None")
                          .vararg().arg<bool>("scaled")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          const at::Tensor &,
                          const c10::optional<at::Tensor> &,
                          bool)>(partial_linear)));
            m.def(utils::FunctionSchemaBuilder("partial_linear").overload("Tensor_scaled")
                          .arg<const at::Tensor &>("input")
                          .arg<const at::Tensor &>("weight")
                          .arg<const c10::optional<at::Tensor> &>("bias", "None")
                          .vararg().arg<bool>("scaled")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          const at::Tensor &,
                          const c10::optional<at::Tensor> &,
                          bool)>(partial_linear)));

            // bilinear
            m.def(utils::FunctionSchemaBuilder("partial_bilinear").overload("MaskedPair_MaskedPair")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("input1")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("input2")
                          .arg<const at::Tensor &>("weight")
                          .arg<const c10::optional<at::Tensor> &>("bias", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          const at::Tensor &,
                          const c10::optional<at::Tensor> &)>(partial_bilinear)));
            m.def(utils::FunctionSchemaBuilder("partial_bilinear").overload("MaskedPair_Tensor")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("input1")
                          .arg<const at::Tensor &>("input2")
                          .arg<const at::Tensor &>("weight")
                          .arg<const c10::optional<at::Tensor> &>("bias", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          const at::Tensor &,
                          const at::Tensor &,
                          const c10::optional<at::Tensor> &)>(partial_bilinear)));
            m.def(utils::FunctionSchemaBuilder("partial_bilinear").overload("Tensor_MaskedPair")
                          .arg<const at::Tensor &>("input1")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("input2")
                          .arg<const at::Tensor &>("weight")
                          .arg<const c10::optional<at::Tensor> &>("bias", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          const at::Tensor &,
                          const c10::optional<at::Tensor> &)>(partial_bilinear)));
            m.def(utils::FunctionSchemaBuilder("partial_bilinear").overload("Tensor_Tensor")
                          .arg<const at::Tensor &>("input1")
                          .arg<const at::Tensor &>("input2")
                          .arg<const at::Tensor &>("weight")
                          .arg<const c10::optional<at::Tensor> &>("bias", "None")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          const at::Tensor &,
                          const at::Tensor &,
                          const c10::optional<at::Tensor> &)>(partial_bilinear)));

            m.def(utils::FunctionSchemaBuilder("partial_bilinear").overload("MaskedPair_MaskedPair_scaled")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("input1")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("input2")
                          .arg<const at::Tensor &>("weight")
                          .arg<const c10::optional<at::Tensor> &>("bias", "None")
                          .vararg().arg<bool>("scaled")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          const at::Tensor &,
                          const c10::optional<at::Tensor> &,
                          bool)>(partial_bilinear)));
            m.def(utils::FunctionSchemaBuilder("partial_bilinear").overload("MaskedPair_Tensor_scaled")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("input1")
                          .arg<const at::Tensor &>("input2")
                          .arg<const at::Tensor &>("weight")
                          .arg<const c10::optional<at::Tensor> &>("bias", "None")
                          .vararg().arg<bool>("scaled")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          const at::Tensor &,
                          const at::Tensor &,
                          const c10::optional<at::Tensor> &,
                          bool)>(partial_bilinear)));
            m.def(utils::FunctionSchemaBuilder("partial_bilinear").overload("Tensor_MaskedPair_scaled")
                          .arg<const at::Tensor &>("input1")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("input2")
                          .arg<const at::Tensor &>("weight")
                          .arg<const c10::optional<at::Tensor> &>("bias", "None")
                          .vararg().arg<bool>("scaled")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          const at::Tensor &,
                          const c10::optional<at::Tensor> &,
                          bool)>(partial_bilinear)));
            m.def(utils::FunctionSchemaBuilder("partial_bilinear").overload("Tensor_Tensor_scaled")
                          .arg<const at::Tensor &>("input1")
                          .arg<const at::Tensor &>("input2")
                          .arg<const at::Tensor &>("weight")
                          .arg<const c10::optional<at::Tensor> &>("bias", "None")
                          .vararg().arg<bool>("scaled")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          const at::Tensor &,
                          const at::Tensor &,
                          const c10::optional<at::Tensor> &,
                          bool)>(partial_bilinear)));
        }
    }
}
