#pragma once

#include <ATen/ATen.h>

namespace partialtorch {
    namespace ops {
        std::tuple<at::Tensor, at::Tensor> _masked_avg_pool1d(
                const at::Tensor &data,
                const at::Tensor &mask,
                at::IntArrayRef kernel_size,
                at::IntArrayRef stride = 1,
                at::IntArrayRef padding = 0,
                bool ceil_mode = false,
                bool count_include_pad = true);

        std::tuple<at::Tensor, at::Tensor> _masked_avg_pool2d(
                const at::Tensor &data,
                const at::Tensor &mask,
                at::IntArrayRef kernel_size,
                at::IntArrayRef stride = 1,
                at::IntArrayRef padding = 0,
                bool ceil_mode = false,
                bool count_include_pad = true);

        std::tuple<at::Tensor, at::Tensor> _masked_avg_pool3d(
                const at::Tensor &data,
                const at::Tensor &mask,
                at::IntArrayRef kernel_size,
                at::IntArrayRef stride = 1,
                at::IntArrayRef padding = 0,
                bool ceil_mode = false,
                bool count_include_pad = true);

        namespace detail {
            at::Tensor __masked_avg_pool1d_backward(
                    const at::Tensor &grad,
                    const at::Tensor &data,
                    const at::Tensor &mask,
                    at::IntArrayRef kernel_size,
                    at::IntArrayRef stride,
                    at::IntArrayRef padding,
                    bool ceil_mode,
                    bool count_include_pad);

            at::Tensor __masked_avg_pool2d_backward(
                    const at::Tensor &grad,
                    const at::Tensor &data,
                    const at::Tensor &mask,
                    at::IntArrayRef kernel_size,
                    at::IntArrayRef stride,
                    at::IntArrayRef padding,
                    bool ceil_mode,
                    bool count_include_pad);

            at::Tensor __masked_avg_pool3d_backward(
                    const at::Tensor &grad,
                    const at::Tensor &data,
                    const at::Tensor &mask,
                    at::IntArrayRef kernel_size,
                    at::IntArrayRef stride,
                    at::IntArrayRef padding,
                    bool ceil_mode,
                    bool count_include_pad);
        }
    }
}
