#include "../masked_adaptive_avg_pool.h"

#include <torch/autograd.h>
#include <torch/types.h>

namespace partialtorch {
    namespace ops {
        namespace {
            class MaskedAdaptiveAveragePool1dFunction
                    : public torch::autograd::Function<MaskedAdaptiveAveragePool1dFunction> {
            public:
                static torch::autograd::variable_list forward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::Variable &data,
                        const torch::autograd::Variable &mask,
                        at::IntArrayRef output_size) {
                    at::AutoDispatchBelowADInplaceOrView g;

                    ctx->save_for_backward({data, mask});
                    ctx->saved_data["output_size"] = output_size;

                    auto output = _masked_adaptive_avg_pool1d(data, mask, output_size);
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
                    auto output_size = ctx->saved_data["output_size"].toIntVector();

                    auto grad_data = detail::__masked_adaptive_avg_pool1d_backward(
                            grad_output[0], data, mask, output_size);

                    return {
                            grad_data,
                            torch::autograd::Variable(),
                            torch::autograd::Variable(),
                    };
                }
            };

            class MaskedAdaptiveAveragePool2dFunction
                    : public torch::autograd::Function<MaskedAdaptiveAveragePool2dFunction> {
            public:
                static torch::autograd::variable_list forward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::Variable &data,
                        const torch::autograd::Variable &mask,
                        at::IntArrayRef output_size) {
                    at::AutoDispatchBelowADInplaceOrView g;

                    ctx->save_for_backward({data, mask});
                    ctx->saved_data["output_size"] = output_size;

                    auto output = _masked_adaptive_avg_pool2d(data, mask, output_size);
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
                    auto output_size = ctx->saved_data["output_size"].toIntVector();

                    auto grad_data = detail::__masked_adaptive_avg_pool2d_backward(
                            grad_output[0], data, mask, output_size);

                    return {
                            grad_data,
                            torch::autograd::Variable(),
                            torch::autograd::Variable(),
                    };
                }
            };

            class MaskedAdaptiveAveragePool3dFunction
                    : public torch::autograd::Function<MaskedAdaptiveAveragePool3dFunction> {
            public:
                static torch::autograd::variable_list forward(
                        torch::autograd::AutogradContext *ctx,
                        const torch::autograd::Variable &data,
                        const torch::autograd::Variable &mask,
                        at::IntArrayRef output_size) {
                    at::AutoDispatchBelowADInplaceOrView g;

                    ctx->save_for_backward({data, mask});
                    ctx->saved_data["output_size"] = output_size;

                    auto output = _masked_adaptive_avg_pool3d(data, mask, output_size);
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
                    auto output_size = ctx->saved_data["output_size"].toIntVector();

                    auto grad_data = detail::__masked_adaptive_avg_pool3d_backward(
                            grad_output[0], data, mask, output_size);

                    return {
                            grad_data,
                            torch::autograd::Variable(),
                            torch::autograd::Variable(),
                    };
                }
            };

            std::tuple<at::Tensor, at::Tensor> _masked_adaptive_avg_pool1d_autograd(
                    const at::Tensor &data,
                    const at::Tensor &mask,
                    at::IntArrayRef output_size) {
                auto result = MaskedAdaptiveAveragePool1dFunction::apply(data, mask, output_size);
                return std::make_tuple(result[0], result[1]);
            }

            std::tuple<at::Tensor, at::Tensor> _masked_adaptive_avg_pool2d_autograd(
                    const at::Tensor &data,
                    const at::Tensor &mask,
                    at::IntArrayRef output_size) {
                auto result = MaskedAdaptiveAveragePool2dFunction::apply(data, mask, output_size);
                return std::make_tuple(result[0], result[1]);
            }

            std::tuple<at::Tensor, at::Tensor> _masked_adaptive_avg_pool3d_autograd(
                    const at::Tensor &data,
                    const at::Tensor &mask,
                    at::IntArrayRef output_size) {
                auto result = MaskedAdaptiveAveragePool3dFunction::apply(data, mask, output_size);
                return std::make_tuple(result[0], result[1]);
            }
        } // namespace

        TORCH_LIBRARY_IMPL(partialtorch, Autograd, m) {
            m.impl(
                    TORCH_SELECTIVE_NAME("partialtorch::_masked_adaptive_avg_pool1d"),
                    TORCH_FN(_masked_adaptive_avg_pool1d_autograd));
            m.impl(
                    TORCH_SELECTIVE_NAME("partialtorch::_masked_adaptive_avg_pool2d"),
                    TORCH_FN(_masked_adaptive_avg_pool2d_autograd));
            m.impl(
                    TORCH_SELECTIVE_NAME("partialtorch::_masked_adaptive_avg_pool3d"),
                    TORCH_FN(_masked_adaptive_avg_pool3d_autograd));
        }
    }
}
