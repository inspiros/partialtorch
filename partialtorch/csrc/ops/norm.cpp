#include <ATen/ATen.h>
#include <torch/library.h>

#include "../MaskedPair.h"
#include "reduction.h"
#include "utils/mask_utils.h"
#include "utils/schema_utils.h"

namespace partialtorch {
    namespace ops {
        namespace impl {
            template<typename input_T>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> batch_norm_impl(
                    const input_T &input,
                    const c10::optional<at::Tensor> &weight,
                    const c10::optional<at::Tensor> &bias,
                    const c10::optional<at::Tensor> &running_mean,
                    const c10::optional<at::Tensor> &running_var,
                    bool training,
                    double momentum,
                    double eps) {
                auto data = utils::get_data(input);
                auto mask = utils::get_mask<true>(input);
                bool feats_only = data.ndimension() == 2;
                if (feats_only) {
                    data = data.unsqueeze(-1);
                    if (mask.has_value())
                        mask = mask->unsqueeze(-1);
                }

                std::vector<int64_t> running_shape;
                running_shape.reserve(data.ndimension());
                running_shape.emplace_back(1);
                running_shape.emplace_back(data.size(1));
                for (const auto _: c10::irange(data.ndimension() - 2))
                    running_shape.emplace_back(1);

                at::Tensor mean, var;
                if (!training) {
                    mean = running_mean->view(running_shape);
                    var = running_var->view(running_shape);
                } else {
                    std::vector<int64_t> norm_dim;
                    norm_dim.reserve(data.ndimension() - 1);
                    norm_dim.emplace_back(0);
                    for (const auto d: c10::irange(2, data.ndimension()))
                        norm_dim.emplace_back(d);

                    auto mean_pair = partialtorch::ops::mean(masked_pair(data, mask), norm_dim, true);
                    mean = mean_pair->data_;
                    auto running_mask = mean_pair->mask_;
                    var = partialtorch::ops::mean(masked_pair((data - mean).pow(2), mask), norm_dim, true)->data_;
                    if (running_mean.has_value() || running_var.has_value()) {
                        if (running_mask.has_value())
                            running_mask->squeeze_();
                        if (running_mean.has_value()) {
                            if (running_mask.has_value())
                                running_mean->index_put_(
                                        {running_mask.value()},
                                        (1.0 - momentum) * running_mean->index({running_mask.value()}) +
                                        momentum * mean.squeeze().index({running_mask.value()}));
                            else
                                running_mean->copy_((1.0 - momentum) * running_mean.value() +
                                                    momentum * mean.squeeze());
                        }
                        if (running_var.has_value()) {
                            if (running_mask.has_value())
                                running_var->index_put_(
                                        {running_mask.value()},
                                        (1.0 - momentum) * running_var->index({running_mask.value()}) +
                                        momentum * var.squeeze().index({running_mask.value()}));
                            else
                                running_var->copy_((1.0 - momentum) * running_var.value() +
                                                   momentum * var.squeeze());
                        }
                    }
                }

                auto x_hat = ((data - mean) / at::sqrt(var + eps));  // * mask.value()
                // afine
                if (weight.has_value()) {  // gamma
                    x_hat.mul_(weight->view(running_shape));
                }
                if (bias.has_value()) {  // beta
                    x_hat.add_(bias->view(running_shape));
                }

                if (feats_only) {
                    x_hat = x_hat.squeeze(-1);
                    if (mask.has_value())
                        mask = mask->squeeze(-1);
                }
                return masked_pair(x_hat, mask);
            }
        }

        c10::intrusive_ptr<TensorMaskedPair> batch_norm(
                const_intrusive_ptr_arg_t<TensorMaskedPair> input,
                const c10::optional<at::Tensor> &weight,
                const c10::optional<at::Tensor> &bias,
                const c10::optional<at::Tensor> &running_mean,
                const c10::optional<at::Tensor> &running_var,
                bool training,
                double momentum,
                double eps) {
            return impl::batch_norm_impl(input, weight, bias, running_mean, running_var, training, momentum, eps);
        }

        c10::intrusive_ptr<TensorMaskedPair> batch_norm(
                const at::Tensor &input,
                const c10::optional<at::Tensor> &weight,
                const c10::optional<at::Tensor> &bias,
                const c10::optional<at::Tensor> &running_mean,
                const c10::optional<at::Tensor> &running_var,
                bool training,
                double momentum,
                double eps) {
            return impl::batch_norm_impl(input, weight, bias, running_mean, running_var, training, momentum, eps);
        }

        TORCH_LIBRARY_FRAGMENT(partialtorch, m) {
            m.def(utils::FunctionSchemaBuilder("batch_norm").overload("MaskedPair")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("input")
                          .arg<const c10::optional<at::Tensor> &>("weight")
                          .arg<const c10::optional<at::Tensor> &>("bias")
                          .arg<const c10::optional<at::Tensor> &>("running_mean")
                          .arg<const c10::optional<at::Tensor> &>("running_var")
                          .arg<bool>("training")
                          .arg<double>("momentum")
                          .arg<double>("eps")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          const c10::optional<at::Tensor> &,
                          const c10::optional<at::Tensor> &,
                          const c10::optional<at::Tensor> &,
                          const c10::optional<at::Tensor> &,
                          bool, double, double)>(batch_norm)));
            m.def(utils::FunctionSchemaBuilder("batch_norm").overload("Tensor")
                          .arg<const at::Tensor &>("input")
                          .arg<const c10::optional<at::Tensor> &>("weight")
                          .arg<const c10::optional<at::Tensor> &>("bias")
                          .arg<const c10::optional<at::Tensor> &>("running_mean")
                          .arg<const c10::optional<at::Tensor> &>("running_var")
                          .arg<bool>("training")
                          .arg<double>("momentum")
                          .arg<double>("eps")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          const c10::optional<at::Tensor> &,
                          const c10::optional<at::Tensor> &,
                          const c10::optional<at::Tensor> &,
                          const c10::optional<at::Tensor> &,
                          bool, double, double)>(batch_norm)));
        }
    }
}
