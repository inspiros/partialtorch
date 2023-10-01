#include <torch/types.h>

#include "izero_div.h"

namespace partialtorch {
    namespace ops {
        namespace utils {
            at::Tensor izero_div(
                    const at::Tensor &self,
                    const at::Tensor &other) {
                static auto op = c10::Dispatcher::singleton()
                                         .findSchemaOrThrow("partialtorch::_izero_div", "")
                                         .typed < at::Tensor(
                const at::Tensor &, const at::Tensor &)>();
                return op.call(
                        self,
                        other);
            }

            at::Tensor izero_ldiv(
                    const at::Tensor &self,
                    const at::Tensor &other) {
                static auto op = c10::Dispatcher::singleton()
                                         .findSchemaOrThrow("partialtorch::_izero_ldiv", "")
                                         .typed < at::Tensor(
                const at::Tensor &, const at::Tensor &)>();
                return op.call(
                        self,
                        other);
            }

            at::Tensor izero_div(
                    const at::Tensor &self,
                    const at::Scalar &other) {
                static auto op = c10::Dispatcher::singleton()
                                         .findSchemaOrThrow("partialtorch::_izero_div", "Scalar")
                                         .typed < at::Tensor(
                const at::Tensor &, const at::Scalar &)>();
                return op.call(
                        self,
                        other);
            }

            at::Tensor izero_ldiv(
                    const at::Tensor &self,
                    const at::Scalar &other) {
                static auto op = c10::Dispatcher::singleton()
                                         .findSchemaOrThrow("partialtorch::_izero_ldiv", "Scalar")
                                         .typed < at::Tensor(
                const at::Tensor &, const at::Scalar &)>();
                return op.call(
                        self,
                        other);
            }

            at::Tensor izero_div(
                    const at::Scalar &self,
                    const at::Tensor &other) {
                static auto op = c10::Dispatcher::singleton()
                                         .findSchemaOrThrow("partialtorch::_izero_div", "rScalar")
                                         .typed < at::Tensor(
                const at::Scalar &, const at::Tensor &)>();
                return op.call(
                        self,
                        other);
            }

            at::Tensor izero_ldiv(
                    const at::Scalar &self,
                    const at::Tensor &other) {
                static auto op = c10::Dispatcher::singleton()
                                         .findSchemaOrThrow("partialtorch::_izero_ldiv", "rScalar")
                                         .typed < at::Tensor(
                const at::Scalar &, const at::Tensor &)>();
                return op.call(
                        self,
                        other);
            }

            at::Tensor izero_div_(
                    at::Tensor &self,
                    const at::Tensor &other) {
                static auto op = c10::Dispatcher::singleton()
                                         .findSchemaOrThrow("partialtorch::_izero_div_", "")
                                         .typed < at::Tensor(at::Tensor & ,
                const at::Tensor &)>();
                return op.call(
                        self,
                        other);
            }

            at::Tensor izero_ldiv_(
                    at::Tensor &self,
                    const at::Tensor &other) {
                static auto op = c10::Dispatcher::singleton()
                                         .findSchemaOrThrow("partialtorch::_izero_ldiv_", "")
                                         .typed < at::Tensor(at::Tensor & ,
                const at::Tensor &)>();
                return op.call(
                        self,
                        other);
            }

            at::Tensor izero_div_(
                    at::Tensor &self,
                    const at::Scalar &other) {
                static auto op = c10::Dispatcher::singleton()
                                         .findSchemaOrThrow("partialtorch::_izero_div_", "Scalar")
                                         .typed < at::Tensor(at::Tensor & ,
                const at::Scalar &)>();
                return op.call(
                        self,
                        other);
            }

            at::Tensor izero_ldiv_(
                    at::Tensor &self,
                    const at::Scalar &other) {
                static auto op = c10::Dispatcher::singleton()
                                         .findSchemaOrThrow("partialtorch::_izero_ldiv_", "Scalar")
                                         .typed < at::Tensor(at::Tensor & ,
                const at::Scalar &)>();
                return op.call(
                        self,
                        other);
            }

            namespace detail {
                std::tuple<at::Tensor, at::Tensor> _izero_div_backward(
                        const at::Tensor &grad,
                        const at::Tensor &self,
                        const at::Tensor &other) {
                    using signature = std::tuple<at::Tensor, at::Tensor>(const at::Tensor &,
                                                                         const at::Tensor &,
                                                                         const at::Tensor &);
                    static auto op =
                            c10::Dispatcher::singleton()
                                    .findSchemaOrThrow("partialtorch::__izero_div_backward", "")
                                    .typed<signature>();
                    return op.call(
                            grad,
                            self,
                            other);
                }

                std::tuple<at::Tensor, at::Tensor> _izero_ldiv_backward(
                        const at::Tensor &grad,
                        const at::Tensor &self,
                        const at::Tensor &other) {
                    using signature = std::tuple<at::Tensor, at::Tensor>(const at::Tensor &,
                                                                         const at::Tensor &,
                                                                         const at::Tensor &);
                    static auto op =
                            c10::Dispatcher::singleton()
                                    .findSchemaOrThrow("partialtorch::__izero_ldiv_backward", "")
                                    .typed<signature>();
                    return op.call(
                            grad,
                            self,
                            other);
                }

                at::Tensor _izero_div_backward(
                        const at::Tensor &grad,
                        const at::Tensor &self,
                        const at::Scalar &other) {
                    using signature = at::Tensor(const at::Tensor &,
                                                 const at::Tensor &,
                                                 const at::Scalar &);
                    static auto op =
                            c10::Dispatcher::singleton()
                                    .findSchemaOrThrow("partialtorch::__izero_div_backward", "Scalar")
                                    .typed<signature>();
                    return op.call(
                            grad,
                            self,
                            other);
                }

                at::Tensor _izero_ldiv_backward(
                        const at::Tensor &grad,
                        const at::Tensor &self,
                        const at::Scalar &other) {
                    using signature = at::Tensor(const at::Tensor &,
                                                 const at::Tensor &,
                                                 const at::Scalar &);
                    static auto op =
                            c10::Dispatcher::singleton()
                                    .findSchemaOrThrow("partialtorch::__izero_ldiv_backward", "Scalar")
                                    .typed<signature>();
                    return op.call(
                            grad,
                            self,
                            other);
                }

                at::Tensor _izero_div_backward(
                        const at::Tensor &grad,
                        const at::Scalar &self,
                        const at::Tensor &other) {
                    using signature = at::Tensor(const at::Tensor &,
                                                 const at::Scalar &,
                                                 const at::Tensor &);
                    static auto op =
                            c10::Dispatcher::singleton()
                                    .findSchemaOrThrow("partialtorch::__izero_div_backward", "rScalar")
                                    .typed<signature>();
                    return op.call(
                            grad,
                            self,
                            other);
                }

                at::Tensor _izero_ldiv_backward(
                        const at::Tensor &grad,
                        const at::Scalar &self,
                        const at::Tensor &other) {
                    using signature = at::Tensor(const at::Tensor &,
                                                 const at::Scalar &,
                                                 const at::Tensor &);
                    static auto op =
                            c10::Dispatcher::singleton()
                                    .findSchemaOrThrow("partialtorch::__izero_ldiv_backward", "rScalar")
                                    .typed<signature>();
                    return op.call(
                            grad,
                            self,
                            other);
                }

                std::tuple<at::Tensor, at::Tensor> _izero_div__backward(
                        const at::Tensor &grad,
                        const at::Tensor &self,
                        const at::Tensor &other) {
                    using signature = std::tuple<at::Tensor, at::Tensor>(const at::Tensor &,
                                                                         const at::Tensor &,
                                                                         const at::Tensor &);
                    static auto op =
                            c10::Dispatcher::singleton()
                                    .findSchemaOrThrow("partialtorch::__izero_div__backward", "")
                                    .typed<signature>();
                    return op.call(
                            grad,
                            self,
                            other);
                }

                std::tuple<at::Tensor, at::Tensor> _izero_ldiv__backward(
                        const at::Tensor &grad,
                        const at::Tensor &self,
                        const at::Tensor &other) {
                    using signature = std::tuple<at::Tensor, at::Tensor>(const at::Tensor &,
                                                                         const at::Tensor &,
                                                                         const at::Tensor &);
                    static auto op =
                            c10::Dispatcher::singleton()
                                    .findSchemaOrThrow("partialtorch::__izero_ldiv__backward", "")
                                    .typed<signature>();
                    return op.call(
                            grad,
                            self,
                            other);
                }

                at::Tensor _izero_div__backward(
                        const at::Tensor &grad,
                        const at::Tensor &self,
                        const at::Scalar &other) {
                    using signature = at::Tensor(const at::Tensor &,
                                                 const at::Tensor &,
                                                 const at::Scalar &);
                    static auto op =
                            c10::Dispatcher::singleton()
                                    .findSchemaOrThrow("partialtorch::__izero_div__backward", "Scalar")
                                    .typed<signature>();
                    return op.call(
                            grad,
                            self,
                            other);
                }

                at::Tensor _izero_ldiv__backward(
                        const at::Tensor &grad,
                        const at::Tensor &self,
                        const at::Scalar &other) {
                    using signature = at::Tensor(const at::Tensor &,
                                                 const at::Tensor &,
                                                 const at::Scalar &);
                    static auto op =
                            c10::Dispatcher::singleton()
                                    .findSchemaOrThrow("partialtorch::__izero_ldiv__backward", "Scalar")
                                    .typed<signature>();
                    return op.call(
                            grad,
                            self,
                            other);
                }
            } // namespace detail

            TORCH_LIBRARY_FRAGMENT(partialtorch, m) {
                m.def(TORCH_SELECTIVE_SCHEMA(
                              "partialtorch::_izero_div(Tensor input, Tensor other) -> Tensor"));
                m.def(TORCH_SELECTIVE_SCHEMA(
                              "partialtorch::_izero_ldiv(Tensor input, Tensor other) -> Tensor"));
                m.def(TORCH_SELECTIVE_SCHEMA(
                              "partialtorch::_izero_div.Scalar(Tensor input, Scalar other) -> Tensor"));
                m.def(TORCH_SELECTIVE_SCHEMA(
                              "partialtorch::_izero_ldiv.Scalar(Tensor input, Scalar other) -> Tensor"));
                m.def(TORCH_SELECTIVE_SCHEMA(
                              "partialtorch::_izero_div.rScalar(Scalar input, Tensor other) -> Tensor"));
                m.def(TORCH_SELECTIVE_SCHEMA(
                              "partialtorch::_izero_ldiv.rScalar(Scalar input, Tensor other) -> Tensor"));
                m.def(TORCH_SELECTIVE_SCHEMA(
                              "partialtorch::_izero_div_(Tensor(a) input, Tensor other) -> Tensor(a)"));
                m.def(TORCH_SELECTIVE_SCHEMA(
                              "partialtorch::_izero_ldiv_(Tensor(a) input, Tensor other) -> Tensor(a)"));
                m.def(TORCH_SELECTIVE_SCHEMA(
                              "partialtorch::_izero_div_.Scalar(Tensor(a) input, Scalar other) -> Tensor(a)"));
                m.def(TORCH_SELECTIVE_SCHEMA(
                              "partialtorch::_izero_ldiv_.Scalar(Tensor(a) input, Scalar other) -> Tensor(a)"));
                m.def(TORCH_SELECTIVE_SCHEMA(
                              "partialtorch::__izero_div_backward(Tensor grad, Tensor self, Tensor other) -> (Tensor grad_self, Tensor grad_other)"));
                m.def(TORCH_SELECTIVE_SCHEMA(
                              "partialtorch::__izero_ldiv_backward(Tensor grad, Tensor self, Tensor other) -> (Tensor grad_self, Tensor grad_other)"));
                m.def(TORCH_SELECTIVE_SCHEMA(
                              "partialtorch::__izero_div_backward.Scalar(Tensor grad, Tensor self, Scalar other) -> Tensor grad_self"));
                m.def(TORCH_SELECTIVE_SCHEMA(
                              "partialtorch::__izero_ldiv_backward.Scalar(Tensor grad, Tensor self, Scalar other) -> Tensor grad_self"));
                m.def(TORCH_SELECTIVE_SCHEMA(
                              "partialtorch::__izero_div_backward.rScalar(Tensor grad, Scalar self, Tensor other) -> Tensor grad_other"));
                m.def(TORCH_SELECTIVE_SCHEMA(
                              "partialtorch::__izero_ldiv_backward.rScalar(Tensor grad, Scalar self, Tensor other) -> Tensor grad_other"));
                m.def(TORCH_SELECTIVE_SCHEMA(
                              "partialtorch::__izero_div__backward(Tensor grad, Tensor self, Tensor other) -> (Tensor grad_self, Tensor grad_other)"));
                m.def(TORCH_SELECTIVE_SCHEMA(
                              "partialtorch::__izero_ldiv__backward(Tensor grad, Tensor self, Tensor other) -> (Tensor grad_self, Tensor grad_other)"));
                m.def(TORCH_SELECTIVE_SCHEMA(
                              "partialtorch::__izero_div__backward.Scalar(Tensor grad, Tensor self, Scalar other) -> Tensor grad_self"));
                m.def(TORCH_SELECTIVE_SCHEMA(
                              "partialtorch::__izero_ldiv__backward.Scalar(Tensor grad, Tensor self, Scalar other) -> Tensor grad_self"));
            }
        } // namespace utils
    } // namespace ops
} // namespace partialtorch
