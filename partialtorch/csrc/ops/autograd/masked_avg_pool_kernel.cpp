#include "../masked_avg_pool.h"

#include <torch/autograd.h>
#include <torch/types.h>

namespace partialtorch {
    namespace ops {
        namespace {
            class MaskedAveragePool1dFunction
                    : public torch::autograd::Function<MaskedAveragePool1dFunction> {
            public:
                static torch::autograd::variable_list forward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::Variable &data,
                        const torch::autograd::Variable &mask,
                        at::IntArrayRef kernel_size,
                        at::IntArrayRef stride,
                        at::IntArrayRef padding,
                        bool ceil_mode,
                        bool count_include_pad) {
                    at::AutoDispatchBelowADInplaceOrView g;

                    ctx->save_for_backward({data, mask});
                    ctx->saved_data["kernel_size"] = kernel_size;
                    ctx->saved_data["stride"] = stride;
                    ctx->saved_data["padding"] = padding;
                    ctx->saved_data["ceil_mode"] = ceil_mode;
                    ctx->saved_data["count_include_pad"] = count_include_pad;

                    auto output = _masked_avg_pool1d(data, mask, kernel_size, stride, padding,
                                                     ceil_mode, count_include_pad);
                    auto output_data = std::get<0>(output);
                    auto output_mask = std::get<1>(output);

                    return {
                        output_data,
                        output_mask,
                    };
                }

                static torch::autograd::variable_list backward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::variable_list &grad_output) {
                    auto saved = ctx->get_saved_variables();
                    auto data = saved[0];
                    auto mask = saved[1];
                    auto kernel_size = ctx->saved_data["kernel_size"].toIntVector();
                    auto stride = ctx->saved_data["stride"].toIntVector();
                    auto padding = ctx->saved_data["padding"].toIntVector();
                    auto ceil_mode = ctx->saved_data["ceil_mode"].toBool();
                    auto count_include_pad = ctx->saved_data["count_include_pad"].toBool();

                    auto grad_data = detail::__masked_avg_pool1d_backward(
                            grad_output[0],
                            data, mask, kernel_size, stride, padding,
                            ceil_mode, count_include_pad);

                    return {
                            grad_data,
                            torch::autograd::Variable(),
                            torch::autograd::Variable(),
                            torch::autograd::Variable(),
                            torch::autograd::Variable(),
                            torch::autograd::Variable(),
                            torch::autograd::Variable(),
                    };
                }
            };

            class MaskedAveragePool2dFunction
                    : public torch::autograd::Function<MaskedAveragePool2dFunction> {
            public:
                static torch::autograd::variable_list forward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::Variable &data,
                        const torch::autograd::Variable &mask,
                        at::IntArrayRef kernel_size,
                        at::IntArrayRef stride,
                        at::IntArrayRef padding,
                        bool ceil_mode,
                        bool count_include_pad) {
                    at::AutoDispatchBelowADInplaceOrView g;

                    ctx->save_for_backward({data, mask});
                    ctx->saved_data["kernel_size"] = kernel_size;
                    ctx->saved_data["stride"] = stride;
                    ctx->saved_data["padding"] = padding;
                    ctx->saved_data["ceil_mode"] = ceil_mode;
                    ctx->saved_data["count_include_pad"] = count_include_pad;

                    auto output = _masked_avg_pool2d(data, mask, kernel_size, stride, padding,
                                                     ceil_mode, count_include_pad);
                    auto output_data = std::get<0>(output);
                    auto output_mask = std::get<1>(output);

                    return {
                        output_data,
                        output_mask,
                    };
                }

                static torch::autograd::variable_list backward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::variable_list &grad_output) {
                    auto saved = ctx->get_saved_variables();
                    auto data = saved[0];
                    auto mask = saved[1];
                    auto kernel_size = ctx->saved_data["kernel_size"].toIntVector();
                    auto stride = ctx->saved_data["stride"].toIntVector();
                    auto padding = ctx->saved_data["padding"].toIntVector();
                    auto ceil_mode = ctx->saved_data["ceil_mode"].toBool();
                    auto count_include_pad = ctx->saved_data["count_include_pad"].toBool();

                    auto grad_data = detail::__masked_avg_pool2d_backward(
                            grad_output[0],
                            data, mask, kernel_size, stride, padding,
                            ceil_mode, count_include_pad);

                    return {
                            grad_data,
                            torch::autograd::Variable(),
                            torch::autograd::Variable(),
                            torch::autograd::Variable(),
                            torch::autograd::Variable(),
                            torch::autograd::Variable(),
                            torch::autograd::Variable(),
                    };
                }
            };

            class MaskedAveragePool3dFunction
                    : public torch::autograd::Function<MaskedAveragePool3dFunction> {
            public:
                static torch::autograd::variable_list forward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::Variable &data,
                        const torch::autograd::Variable &mask,
                        at::IntArrayRef kernel_size,
                        at::IntArrayRef stride,
                        at::IntArrayRef padding,
                        bool ceil_mode,
                        bool count_include_pad) {
                    at::AutoDispatchBelowADInplaceOrView g;

                    ctx->save_for_backward({data, mask});
                    ctx->saved_data["kernel_size"] = kernel_size;
                    ctx->saved_data["stride"] = stride;
                    ctx->saved_data["padding"] = padding;
                    ctx->saved_data["ceil_mode"] = ceil_mode;
                    ctx->saved_data["count_include_pad"] = count_include_pad;

                    auto output = _masked_avg_pool3d(data, mask, kernel_size, stride, padding,
                                                     ceil_mode, count_include_pad);
                    auto output_data = std::get<0>(output);
                    auto output_mask = std::get<1>(output);

                    return {
                        output_data,
                        output_mask,
                    };
                }

                static torch::autograd::variable_list backward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::variable_list &grad_output) {
                    auto saved = ctx->get_saved_variables();
                    auto data = saved[0];
                    auto mask = saved[1];
                    auto kernel_size = ctx->saved_data["kernel_size"].toIntVector();
                    auto stride = ctx->saved_data["stride"].toIntVector();
                    auto padding = ctx->saved_data["padding"].toIntVector();
                    auto ceil_mode = ctx->saved_data["ceil_mode"].toBool();
                    auto count_include_pad = ctx->saved_data["count_include_pad"].toBool();

                    auto grad_data = detail::__masked_avg_pool3d_backward(
                            grad_output[0],
                            data, mask, kernel_size, stride, padding,
                            ceil_mode, count_include_pad);

                    return {
                            grad_data,
                            torch::autograd::Variable(),
                            torch::autograd::Variable(),
                            torch::autograd::Variable(),
                            torch::autograd::Variable(),
                            torch::autograd::Variable(),
                            torch::autograd::Variable(),
                    };
                }
            };

            std::tuple<at::Tensor, at::Tensor> _masked_avg_pool1d_autograd(
                    const at::Tensor &data,
                    const at::Tensor &mask,
                    at::IntArrayRef kernel_size,
                    at::IntArrayRef stride,
                    at::IntArrayRef padding,
                    bool ceil_mode,
                    bool count_include_pad) {
                auto result = MaskedAveragePool1dFunction::apply(data, mask, kernel_size, stride, padding,
                                                                 ceil_mode, count_include_pad);
                return std::make_tuple(result[0], result[1]);
            }

            std::tuple<at::Tensor, at::Tensor> _masked_avg_pool2d_autograd(
                    const at::Tensor &data,
                    const at::Tensor &mask,
                    at::IntArrayRef kernel_size,
                    at::IntArrayRef stride,
                    at::IntArrayRef padding,
                    bool ceil_mode,
                    bool count_include_pad) {
                auto result = MaskedAveragePool2dFunction::apply(data, mask, kernel_size, stride, padding,
                                                                 ceil_mode, count_include_pad);
                return std::make_tuple(result[0], result[1]);
            }

            std::tuple<at::Tensor, at::Tensor> _masked_avg_pool3d_autograd(
                    const at::Tensor &data,
                    const at::Tensor &mask,
                    at::IntArrayRef kernel_size,
                    at::IntArrayRef stride,
                    at::IntArrayRef padding,
                    bool ceil_mode,
                    bool count_include_pad) {
                auto result = MaskedAveragePool3dFunction::apply(data, mask, kernel_size, stride, padding,
                                                                 ceil_mode, count_include_pad);
                return std::make_tuple(result[0], result[1]);
            }
        } // namespace

        TORCH_LIBRARY_IMPL(partialtorch, Autograd, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("partialtorch::_masked_avg_pool1d"),
                    TORCH_FN(_masked_avg_pool1d_autograd));
            m.impl(
                    TORCH_SELECTIVE_NAME("partialtorch::_masked_avg_pool2d"),
                    TORCH_FN(_masked_avg_pool2d_autograd));
            m.impl(
                    TORCH_SELECTIVE_NAME("partialtorch::_masked_avg_pool3d"),
                    TORCH_FN(_masked_avg_pool3d_autograd));
        }
    }
}
