#pragma once

#include <ATen/ATen.h>

namespace partialtorch {
    namespace ops {
        namespace utils {
            at::Tensor izero_div(
                    const at::Tensor &self,
                    const at::Tensor &other);

            at::Tensor izero_ldiv(
                    const at::Tensor &self,
                    const at::Tensor &other);

            at::Tensor izero_div(
                    const at::Tensor &self,
                    const at::Scalar &other);

            at::Tensor izero_ldiv(
                    const at::Tensor &self,
                    const at::Scalar &other);

            at::Tensor izero_div(
                    const at::Scalar &self,
                    const at::Tensor &other);

            at::Tensor izero_ldiv(
                    const at::Scalar &self,
                    const at::Tensor &other);

            at::Tensor izero_div_(
                    at::Tensor &self,
                    const at::Tensor &other);

            at::Tensor izero_ldiv_(
                    at::Tensor &self,
                    const at::Tensor &other);

            at::Tensor izero_div_(
                    at::Tensor &self,
                    const at::Scalar &other);

            at::Tensor izero_ldiv_(
                    at::Tensor &self,
                    const at::Scalar &other);

            namespace detail {
                std::tuple<at::Tensor, at::Tensor> _izero_div_backward(
                        const at::Tensor &grad_output,
                        const at::Tensor &self,
                        const at::Tensor &other);

                std::tuple<at::Tensor, at::Tensor> _izero_ldiv_backward(
                        const at::Tensor &grad_output,
                        const at::Tensor &self,
                        const at::Tensor &other);

                at::Tensor _izero_div_backward(
                        const at::Tensor &grad_output,
                        const at::Tensor &self,
                        const at::Scalar &other);

                at::Tensor _izero_ldiv_backward(
                        const at::Tensor &grad_output,
                        const at::Tensor &self,
                        const at::Scalar &other);

                at::Tensor _izero_div_backward(
                        const at::Tensor &grad_output,
                        const at::Scalar &self,
                        const at::Tensor &other);

                at::Tensor _izero_ldiv_backward(
                        const at::Tensor &grad_output,
                        const at::Scalar &self,
                        const at::Tensor &other);

                std::tuple<at::Tensor, at::Tensor> _izero_div__backward(
                        const at::Tensor &grad_output,
                        const at::Tensor &self,
                        const at::Tensor &other);

                std::tuple<at::Tensor, at::Tensor> _izero_ldiv__backward(
                        const at::Tensor &grad_output,
                        const at::Tensor &self,
                        const at::Tensor &other);

                at::Tensor _izero_div__backward(
                        const at::Tensor &grad_output,
                        const at::Tensor &self,
                        const at::Scalar &other);

                at::Tensor _izero_ldiv__backward(
                        const at::Tensor &grad_output,
                        const at::Tensor &self,
                        const at::Scalar &other);
            }
        }
    }
}
