#include "masked_avg_pool.h"

#include <torch/types.h>

namespace partialtorch {
    namespace ops {
        std::tuple<at::Tensor, at::Tensor> _masked_avg_pool1d(
                const at::Tensor &data,
                const at::Tensor &mask,
                at::IntArrayRef kernel_size,
                at::IntArrayRef stride,
                at::IntArrayRef padding,
                bool ceil_mode,
                bool count_include_pad) {
            static auto op = c10::Dispatcher::singleton()
                    .findSchemaOrThrow("partialtorch::_masked_avg_pool1d", "")
                    .typed<decltype(_masked_avg_pool1d)>();
            return op.call(data, mask, kernel_size, stride, padding, ceil_mode, count_include_pad);
        }

        std::tuple<at::Tensor, at::Tensor> _masked_avg_pool2d(
                const at::Tensor &data,
                const at::Tensor &mask,
                at::IntArrayRef kernel_size,
                at::IntArrayRef stride,
                at::IntArrayRef padding,
                bool ceil_mode,
                bool count_include_pad) {
            static auto op = c10::Dispatcher::singleton()
                    .findSchemaOrThrow("partialtorch::_masked_avg_pool2d", "")
                    .typed<decltype(_masked_avg_pool2d)>();
            return op.call(data, mask, kernel_size, stride, padding, ceil_mode, count_include_pad);
        }

        std::tuple<at::Tensor, at::Tensor> _masked_avg_pool3d(
                const at::Tensor &data,
                const at::Tensor &mask,
                at::IntArrayRef kernel_size,
                at::IntArrayRef stride,
                at::IntArrayRef padding,
                bool ceil_mode,
                bool count_include_pad) {
            static auto op = c10::Dispatcher::singleton()
                    .findSchemaOrThrow("partialtorch::_masked_avg_pool3d", "")
                    .typed<decltype(_masked_avg_pool3d)>();
            return op.call(data, mask, kernel_size, stride, padding, ceil_mode, count_include_pad);
        }

        namespace detail {
            at::Tensor __masked_avg_pool1d_backward(
                    const at::Tensor &grad,
                    const at::Tensor &data,
                    const at::Tensor &mask,
                    at::IntArrayRef kernel_size,
                    at::IntArrayRef stride,
                    at::IntArrayRef padding,
                    bool ceil_mode,
                    bool count_include_pad) {
                static auto op =
                        c10::Dispatcher::singleton()
                                .findSchemaOrThrow("partialtorch::__masked_avg_pool1d_backward", "")
                                .typed<decltype(__masked_avg_pool1d_backward)>();
                return op.call(grad, data, mask, kernel_size, stride, padding, ceil_mode, count_include_pad);
            }

            at::Tensor __masked_avg_pool2d_backward(
                    const at::Tensor &grad,
                    const at::Tensor &data,
                    const at::Tensor &mask,
                    at::IntArrayRef kernel_size,
                    at::IntArrayRef stride,
                    at::IntArrayRef padding,
                    bool ceil_mode,
                    bool count_include_pad) {
                static auto op =
                        c10::Dispatcher::singleton()
                                .findSchemaOrThrow("partialtorch::__masked_avg_pool2d_backward", "")
                                .typed<decltype(__masked_avg_pool2d_backward)>();
                return op.call(grad, data, mask, kernel_size, stride, padding, ceil_mode, count_include_pad);
            }

            at::Tensor __masked_avg_pool3d_backward(
                    const at::Tensor &grad,
                    const at::Tensor &data,
                    const at::Tensor &mask,
                    at::IntArrayRef kernel_size,
                    at::IntArrayRef stride,
                    at::IntArrayRef padding,
                    bool ceil_mode,
                    bool count_include_pad) {
                static auto op =
                        c10::Dispatcher::singleton()
                                .findSchemaOrThrow("partialtorch::__masked_avg_pool3d_backward", "")
                                .typed<decltype(__masked_avg_pool3d_backward)>();
                return op.call(grad, data, mask, kernel_size, stride, padding, ceil_mode, count_include_pad);
            }
        }

        TORCH_LIBRARY_FRAGMENT(partialtorch, m) {
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "partialtorch::_masked_avg_pool1d(Tensor data, Tensor mask, "
                          "int[1] kernel_size, int[1] stride=1, int[1] padding=0, "
                          "bool ceil_mode=False, bool count_include_pad=True"
                          ") -> (Tensor output_data, Tensor output_mask)")
            );
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "partialtorch::_masked_avg_pool2d(Tensor data, Tensor mask, "
                          "int[2] kernel_size, int[2] stride=1, int[2] padding=0, "
                          "bool ceil_mode=False, bool count_include_pad=True"
                          ") -> (Tensor output_data, Tensor output_mask)")
            );
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "partialtorch::_masked_avg_pool3d(Tensor data, Tensor mask, "
                          "int[3] kernel_size, int[3] stride=1, int[3] padding=0, "
                          "bool ceil_mode=False, bool count_include_pad=True"
                          ") -> (Tensor output_data, Tensor output_mask)")
            );
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "partialtorch::__masked_avg_pool1d_backward(Tensor grad, Tensor data, Tensor mask, "
                          "int[1] kernel_size, int[1] stride, int[1] padding, "
                          "bool ceil_mode=False, bool count_include_pad=True) -> Tensor")
            );
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "partialtorch::__masked_avg_pool2d_backward(Tensor grad, Tensor data, Tensor mask, "
                          "int[2] kernel_size, int[2] stride, int[2] padding, "
                          "bool ceil_mode=False, bool count_include_pad=True) -> Tensor")
            );
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "partialtorch::__masked_avg_pool3d_backward(Tensor grad, Tensor data, Tensor mask, "
                          "int[3] kernel_size, int[3] stride, int[3] padding, "
                          "bool ceil_mode=False, bool count_include_pad=True) -> Tensor")
            );
        }
    } // namespace ops
} // namespace partialtorch
