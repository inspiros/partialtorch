#include "masked_adaptive_avg_pool.h"

#include <torch/types.h>

namespace partialtorch {
    namespace ops {
        std::tuple<at::Tensor, at::Tensor> _masked_adaptive_avg_pool1d(
                const at::Tensor &data,
                const at::Tensor &mask,
                at::IntArrayRef output_size) {
            static auto op = c10::Dispatcher::singleton()
                    .findSchemaOrThrow("partialtorch::_masked_adaptive_avg_pool1d", "")
                    .typed<decltype(_masked_adaptive_avg_pool1d)>();
            return op.call(data, mask, output_size);
        }

        std::tuple<at::Tensor, at::Tensor> _masked_adaptive_avg_pool2d(
                const at::Tensor &data,
                const at::Tensor &mask,
                at::IntArrayRef output_size) {
            static auto op = c10::Dispatcher::singleton()
                    .findSchemaOrThrow("partialtorch::_masked_adaptive_avg_pool2d", "")
                    .typed<decltype(_masked_adaptive_avg_pool2d)>();
            return op.call(data, mask, output_size);
        }

        std::tuple<at::Tensor, at::Tensor> _masked_adaptive_avg_pool3d(
                const at::Tensor &data,
                const at::Tensor &mask,
                at::IntArrayRef output_size) {
            static auto op = c10::Dispatcher::singleton()
                    .findSchemaOrThrow("partialtorch::_masked_adaptive_avg_pool3d", "")
                    .typed<decltype(_masked_adaptive_avg_pool3d)>();
            return op.call(data, mask, output_size);
        }

        namespace detail {
            at::Tensor __masked_adaptive_avg_pool1d_backward(
                    const at::Tensor &grad,
                    const at::Tensor &data,
                    const at::Tensor &mask,
                    at::IntArrayRef output_size) {
                static auto op =
                        c10::Dispatcher::singleton()
                                .findSchemaOrThrow("partialtorch::__masked_adaptive_avg_pool1d_backward", "")
                                .typed<decltype(__masked_adaptive_avg_pool1d_backward)>();
                return op.call(grad, data, mask, output_size);
            }

            at::Tensor __masked_adaptive_avg_pool2d_backward(
                    const at::Tensor &grad,
                    const at::Tensor &data,
                    const at::Tensor &mask,
                    at::IntArrayRef output_size) {
                static auto op =
                        c10::Dispatcher::singleton()
                                .findSchemaOrThrow("partialtorch::__masked_adaptive_avg_pool2d_backward", "")
                                .typed<decltype(__masked_adaptive_avg_pool2d_backward)>();
                return op.call(grad, data, mask, output_size);
            }

            at::Tensor __masked_adaptive_avg_pool3d_backward(
                    const at::Tensor &grad,
                    const at::Tensor &data,
                    const at::Tensor &mask,
                    at::IntArrayRef output_size) {
                static auto op =
                        c10::Dispatcher::singleton()
                                .findSchemaOrThrow("partialtorch::__masked_adaptive_avg_pool3d_backward", "")
                                .typed<decltype(__masked_adaptive_avg_pool3d_backward)>();
                return op.call(grad, data, mask, output_size);
            }
        }

        TORCH_LIBRARY_FRAGMENT(partialtorch, m) {
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "partialtorch::_masked_adaptive_avg_pool1d(Tensor data, Tensor mask, "
                          "int[1] output_size) -> (Tensor output_data, Tensor output_mask)")
            );
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "partialtorch::_masked_adaptive_avg_pool2d(Tensor data, Tensor mask, "
                          "int[2] output_size) -> (Tensor output_data, Tensor output_mask)")
            );
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "partialtorch::_masked_adaptive_avg_pool3d(Tensor data, Tensor mask, "
                          "int[3] output_size) -> (Tensor output_data, Tensor output_mask)")
            );
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "partialtorch::__masked_adaptive_avg_pool1d_backward(Tensor grad, Tensor data, Tensor mask, "
                          "int[1] output_size) -> Tensor")
            );
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "partialtorch::__masked_adaptive_avg_pool2d_backward(Tensor grad, Tensor data, Tensor mask, "
                          "int[2] output_size) -> Tensor")
            );
            m.def(TORCH_SELECTIVE_SCHEMA(
                          "partialtorch::__masked_adaptive_avg_pool3d_backward(Tensor grad, Tensor data, Tensor mask, "
                          "int[3] output_size) -> Tensor")
            );
        }
    } // namespace ops
} // namespace partialtorch
