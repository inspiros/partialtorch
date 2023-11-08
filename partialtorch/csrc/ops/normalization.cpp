#include <ATen/ATen.h>
#include <torch/library.h>

#include "../MaskedPair.h"
#include "reduction.h"
#include "utils/mask_utils.h"
#include "utils/reduction_utils.h"
#include "utils/schema_utils.h"

namespace partialtorch {
    namespace ops {
        namespace impl {
            enum NormMode {
                BatchNorm, InstanceNorm, LayerNorm, GroupNorm
            };

            template<NormMode mode, typename input_T>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> tracked_norm_impl(
                    const input_T &input,
                    const c10::optional<at::Tensor> &weight,
                    const c10::optional<at::Tensor> &bias,
                    const c10::optional<at::Tensor> &running_mean,
                    const c10::optional<at::Tensor> &running_var,
                    bool training,
                    double momentum,
                    double eps,
                    bool cudnn_enabled) {
                if (!utils::has_tensor_mask(input)) {
                    if constexpr (mode == BatchNorm) {
                        return masked_pair(
                                at::_ops::batch_norm::call(utils::get_data(input), weight, bias,
                                                           running_mean, running_var, training, momentum, eps,
                                                           cudnn_enabled));
                    } else if constexpr (mode == InstanceNorm) {
                        return masked_pair(
                                at::_ops::instance_norm::call(utils::get_data(input), weight, bias,
                                                              running_mean, running_var, training, momentum, eps,
                                                              cudnn_enabled));
                    }
                }

                auto data = utils::get_data(input);
                auto mask = utils::get_mask<true>(input);
                bool feats_only = data.ndimension() == 2;
                if (feats_only) {
                    data = data.unsqueeze(-1);
                    mask = mask->unsqueeze(-1);
                }

                std::vector<int64_t> normalized_shape;
                normalized_shape.reserve(data.ndimension());
                normalized_shape.emplace_back(1);
                normalized_shape.emplace_back(data.size(1));
                for (const auto _: c10::irange(data.ndimension() - 2))
                    normalized_shape.emplace_back(1);

                at::Tensor mean, var;
                if (!training) {
                    mean = running_mean->view(normalized_shape);
                    var = running_var->view(normalized_shape);
                } else {
                    std::vector<int64_t> norm_dim;
                    if constexpr (mode == BatchNorm) {
                        norm_dim.reserve(data.ndimension() - 1);
                        norm_dim.emplace_back(0);
                    } else if constexpr (mode == InstanceNorm) {
                        norm_dim.reserve(data.ndimension() - 2);
                    }
                    for (const auto d: c10::irange(2, data.ndimension()))
                        norm_dim.emplace_back(d);

                    auto input_view = masked_pair(data, mask);
                    auto mean_pair = partialtorch::ops::mean(input_view, norm_dim, true);
                    mean = utils::get_data(mean_pair);
                    auto running_mask = utils::get_tensor_mask(mean_pair);
                    auto x = (data - mean).masked_fill_(mask->logical_not(), 0);
                    auto count = utils::reduction_numel(input_view, norm_dim, true).sub_(1).clamp_min_(0);
                    var = at::sum(x * x.conj(), norm_dim, true).div_(count);
                    if (running_mean.has_value() || running_var.has_value()) {
                        running_mask.squeeze_();
                        if (running_mean.has_value()) {
                            auto updated_running_mean =
                                    (1.0 - momentum) * running_mean->index({running_mask}) +
                                    momentum * mean.squeeze().index({running_mask});
                            if constexpr (mode == InstanceNorm)
                                updated_running_mean = updated_running_mean.mean(0);
                            running_mean->index_put_({running_mask}, updated_running_mean);
                        }
                        if (running_var.has_value()) {
                            running_mask.bitwise_and_(count.view_as(running_mask).ne(0));
                            auto updated_running_var =
                                    (1.0 - momentum) * running_var->index({running_mask}) +
                                    momentum * var.squeeze().index({running_mask});
                            if constexpr (mode == InstanceNorm)
                                updated_running_var = updated_running_var.mean(0);
                            running_var->index_put_({running_mask}, updated_running_var);
                        }
                    }
                }

                auto x_hat = (data - mean) / at::sqrt(var + eps);
                // affine
                if (weight.has_value())  // gamma
                    x_hat.mul_(weight->view(normalized_shape));
                if (bias.has_value())  // beta
                    x_hat.add_(bias->view(normalized_shape));

                if (feats_only) {
                    x_hat = x_hat.squeeze(-1);
                    mask = mask->squeeze(-1);
                }
                return masked_pair(x_hat, mask);
            }

            template<typename input_T>
            static C10_ALWAYS_INLINE void check_layer_norm_inputs(
                    const input_T &input,
                    at::IntArrayRef normalized_shape,
                    const c10::optional<at::Tensor> &weight,
                    const c10::optional<at::Tensor> &bias) {
                auto input_shape = utils::get_data(input).sizes();
                auto input_ndim = input_shape.size();
                auto normalized_ndim = normalized_shape.size();
                TORCH_CHECK(
                        normalized_ndim >= 1,
                        "Expected normalized_shape to be at least 1-dimensional, i.e., ",
                        "containing at least one element, but got normalized_shape = ",
                        normalized_shape)
                TORCH_CHECK(
                        !weight.has_value() || weight->sizes().equals(normalized_shape),
                        "Expected weight to be of same shape as normalized_shape, but got ",
                        "weight of shape ",
                        weight->sizes(),
                        " and normalized_shape = ",
                        normalized_shape)
                TORCH_CHECK(
                        !bias.has_value() || bias->sizes().equals(normalized_shape),
                        "Expected bias to be of same shape as normalized_shape, but got ",
                        "bias of shape ",
                        bias->sizes(),
                        " and normalized_shape = ",
                        normalized_shape)
                TORCH_CHECK(
                        input_ndim >= normalized_ndim &&
                        input_shape.slice(input_ndim - normalized_ndim, input_ndim - 1).equals(normalized_shape),
                        "Given normalized_shape=",
                        normalized_shape,
                        ", expected input with shape [*",
                        std::accumulate(
                                std::begin(normalized_shape), std::end(normalized_shape), std::string(),
                                [](const std::string &ss, const int64_t d) {
                                    return ss + ", " + std::to_string(d);
                                }),
                        "], but got input of size ",
                        input_shape)
            }

            template<typename input_T>
            static C10_ALWAYS_INLINE c10::intrusive_ptr<TensorMaskedPair> layer_norm_impl(
                    const input_T &input,
                    at::SymIntArrayRef normalized_shape,
                    const c10::optional<at::Tensor> &weight,
                    const c10::optional<at::Tensor> &bias,
                    double eps,
                    bool cudnn_enabled) {
                if (!utils::has_tensor_mask(input)) {
                    return masked_pair(
                            at::_ops::layer_norm::call(utils::get_data(input), normalized_shape, weight, bias,
                                                       eps, cudnn_enabled));
                }

                check_layer_norm_inputs(input, c10::asIntArrayRefUnchecked(normalized_shape), weight, bias);
                auto data = utils::get_data(input);
                auto mask = utils::get_mask<true>(input);
                bool feats_only = data.ndimension() == 2;
                if (feats_only) {
                    data = data.unsqueeze(-1);
                    mask = mask->unsqueeze(-1);
                }

                at::Tensor mean, var;
                {
                    std::vector<int64_t> norm_dim;
                    norm_dim.reserve(data.ndimension() - normalized_shape.size());
                    for (const auto d: c10::irange(data.ndimension() - normalized_shape.size(), data.ndimension()))
                        norm_dim.emplace_back(d);

                    auto input_view = masked_pair(data, mask);
                    auto mean_pair = partialtorch::ops::mean(input_view, norm_dim, true);
                    mean = utils::get_data(mean_pair);
                    auto running_mask = utils::get_tensor_mask(mean_pair);
                    auto x = (data - mean).masked_fill_(mask->logical_not(), 0);
                    auto count = utils::reduction_numel(input_view, norm_dim, true).clamp_min_(0);
                    var = at::sum(x * x.conj(), norm_dim, true).div_(count);
                }

                auto x_hat = (data - mean) / at::sqrt(var + eps);
                // affine
                if (weight.has_value())  // gamma
                    x_hat.mul_(weight.value());
                if (bias.has_value())  // beta
                    x_hat.add_(bias.value());

                if (feats_only) {
                    x_hat = x_hat.squeeze(-1);
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
                double eps,
                bool cudnn_enabled) {
            return impl::tracked_norm_impl<impl::BatchNorm>(
                    input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
        }

        c10::intrusive_ptr<TensorMaskedPair> batch_norm(
                const at::Tensor &input,
                const c10::optional<at::Tensor> &weight,
                const c10::optional<at::Tensor> &bias,
                const c10::optional<at::Tensor> &running_mean,
                const c10::optional<at::Tensor> &running_var,
                bool training,
                double momentum,
                double eps,
                bool cudnn_enabled) {
            return impl::tracked_norm_impl<impl::BatchNorm>(
                    input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
        }

        c10::intrusive_ptr<TensorMaskedPair> instance_norm(
                const_intrusive_ptr_arg_t<TensorMaskedPair> input,
                const c10::optional<at::Tensor> &weight,
                const c10::optional<at::Tensor> &bias,
                const c10::optional<at::Tensor> &running_mean,
                const c10::optional<at::Tensor> &running_var,
                bool use_input_stats,
                double momentum,
                double eps,
                bool cudnn_enabled) {
            return impl::tracked_norm_impl<impl::InstanceNorm>(
                    input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled);
        }

        c10::intrusive_ptr<TensorMaskedPair> instance_norm(
                const at::Tensor &input,
                const c10::optional<at::Tensor> &weight,
                const c10::optional<at::Tensor> &bias,
                const c10::optional<at::Tensor> &running_mean,
                const c10::optional<at::Tensor> &running_var,
                bool use_input_stats,
                double momentum,
                double eps,
                bool cudnn_enabled) {
            return impl::tracked_norm_impl<impl::InstanceNorm>(
                    input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled);
        }

        c10::intrusive_ptr<TensorMaskedPair> layer_norm(
                const_intrusive_ptr_arg_t<TensorMaskedPair> input,
                at::SymIntArrayRef normalized_shape,
                const c10::optional<at::Tensor> &weight,
                const c10::optional<at::Tensor> &bias,
                double eps,
                bool cudnn_enabled) {
            return impl::layer_norm_impl(input, normalized_shape, weight, bias, eps, cudnn_enabled);
        }

        c10::intrusive_ptr<TensorMaskedPair> layer_norm(
                const at::Tensor &input,
                at::SymIntArrayRef normalized_shape,
                const c10::optional<at::Tensor> &weight,
                const c10::optional<at::Tensor> &bias,
                double eps,
                bool cudnn_enabled) {
            return impl::layer_norm_impl(input, normalized_shape, weight, bias, eps, cudnn_enabled);
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
                          .arg<bool>("cudnn_enabled")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          const c10::optional<at::Tensor> &,
                          const c10::optional<at::Tensor> &,
                          const c10::optional<at::Tensor> &,
                          const c10::optional<at::Tensor> &,
                          bool, double, double, bool)>(batch_norm)));
            m.def(utils::FunctionSchemaBuilder("batch_norm").overload("Tensor")
                          .arg<const at::Tensor &>("input")
                          .arg<const c10::optional<at::Tensor> &>("weight")
                          .arg<const c10::optional<at::Tensor> &>("bias")
                          .arg<const c10::optional<at::Tensor> &>("running_mean")
                          .arg<const c10::optional<at::Tensor> &>("running_var")
                          .arg<bool>("training")
                          .arg<double>("momentum")
                          .arg<double>("eps")
                          .arg<bool>("cudnn_enabled")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          const c10::optional<at::Tensor> &,
                          const c10::optional<at::Tensor> &,
                          const c10::optional<at::Tensor> &,
                          const c10::optional<at::Tensor> &,
                          bool, double, double, bool)>(batch_norm)));

            m.def(utils::FunctionSchemaBuilder("instance_norm").overload("MaskedPair")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("input")
                          .arg<const c10::optional<at::Tensor> &>("weight")
                          .arg<const c10::optional<at::Tensor> &>("bias")
                          .arg<const c10::optional<at::Tensor> &>("running_mean")
                          .arg<const c10::optional<at::Tensor> &>("running_var")
                          .arg<bool>("use_input_stats")
                          .arg<double>("momentum")
                          .arg<double>("eps")
                          .arg<bool>("cudnn_enabled")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          const c10::optional<at::Tensor> &,
                          const c10::optional<at::Tensor> &,
                          const c10::optional<at::Tensor> &,
                          const c10::optional<at::Tensor> &,
                          bool, double, double, bool)>(instance_norm)));
            m.def(utils::FunctionSchemaBuilder("instance_norm").overload("Tensor")
                          .arg<const at::Tensor &>("input")
                          .arg<const c10::optional<at::Tensor> &>("weight")
                          .arg<const c10::optional<at::Tensor> &>("bias")
                          .arg<const c10::optional<at::Tensor> &>("running_mean")
                          .arg<const c10::optional<at::Tensor> &>("running_var")
                          .arg<bool>("use_input_stats")
                          .arg<double>("momentum")
                          .arg<double>("eps")
                          .arg<bool>("cudnn_enabled")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          const c10::optional<at::Tensor> &,
                          const c10::optional<at::Tensor> &,
                          const c10::optional<at::Tensor> &,
                          const c10::optional<at::Tensor> &,
                          bool, double, double, bool)>(instance_norm)));

            m.def(utils::FunctionSchemaBuilder("layer_norm").overload("MaskedPair")
                          .arg<const_intrusive_ptr_arg_t<TensorMaskedPair>>("input")
                          .arg<c10::SymIntArrayRef, 0>("normalized_shape")
                          .arg<const c10::optional<at::Tensor> &>("weight", "None")
                          .arg<const c10::optional<at::Tensor> &>("bias", "None")
                          .arg<double>("eps", "1e-5")
                          .arg<bool>("cudnn_enabled", "True")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const_intrusive_ptr_arg_t<TensorMaskedPair>,
                          c10::SymIntArrayRef,
                          const c10::optional<at::Tensor> &,
                          const c10::optional<at::Tensor> &,
                          double, bool)>(layer_norm)));
            m.def(utils::FunctionSchemaBuilder("layer_norm").overload("Tensor")
                          .arg<const at::Tensor &>("input")
                          .arg<c10::SymIntArrayRef, 0>("normalized_shape")
                          .arg<const c10::optional<at::Tensor> &>("weight", "None")
                          .arg<const c10::optional<at::Tensor> &>("bias", "None")
                          .arg<double>("eps", "1e-5")
                          .arg<bool>("cudnn_enabled", "True")
                          .ret<TensorMaskedPair>().schema().c_str(),
                  TORCH_FN(static_cast<c10::intrusive_ptr<TensorMaskedPair> (*)(
                          const at::Tensor &,
                          c10::SymIntArrayRef,
                          const c10::optional<at::Tensor> &,
                          const c10::optional<at::Tensor> &,
                          double, bool)>(layer_norm)));
        }
    }
}
