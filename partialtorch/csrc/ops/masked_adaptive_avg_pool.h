#pragma once

#include <ATen/ATen.h>

namespace partialtorch {
    namespace ops {
        std::tuple<at::Tensor, at::Tensor> _masked_adaptive_avg_pool1d(
                const at::Tensor &data,
                const at::Tensor &mask,
                at::IntArrayRef output_size);

        std::tuple<at::Tensor, at::Tensor> _masked_adaptive_avg_pool2d(
                const at::Tensor &data,
                const at::Tensor &mask,
                at::IntArrayRef output_size);

        std::tuple<at::Tensor, at::Tensor> _masked_adaptive_avg_pool3d(
                const at::Tensor &data,
                const at::Tensor &mask,
                at::IntArrayRef output_size);

        namespace detail {
            at::Tensor __masked_adaptive_avg_pool1d_backward(
                    const at::Tensor &grad,
                    const at::Tensor &data,
                    const at::Tensor &mask,
                    at::IntArrayRef output_size);

            at::Tensor __masked_adaptive_avg_pool2d_backward(
                    const at::Tensor &grad,
                    const at::Tensor &data,
                    const at::Tensor &mask,
                    at::IntArrayRef output_size);

            at::Tensor __masked_adaptive_avg_pool3d_backward(
                    const at::Tensor &grad,
                    const at::Tensor &data,
                    const at::Tensor &mask,
                    at::IntArrayRef output_size);
        }
    }
}
