#include <torch/autograd.h>
#include <torch/types.h>

#include "../izero_div.h"

namespace partialtorch {
    namespace ops {
        namespace utils {
            namespace {
                class IZeroDivideFunction
                        : public torch::autograd::Function<IZeroDivideFunction> {
                public:
                    static torch::autograd::Variable forward(
                            torch::autograd::AutogradContext *ctx,
                            const torch::autograd::Variable &self,
                            const torch::autograd::Variable &other) {
                        at::AutoDispatchBelowADInplaceOrView g;
                        auto output = izero_div(
                                self,
                                other);

                        ctx->save_for_backward({self, other});

                        return output;
                    }

                    static torch::autograd::variable_list backward(
                            torch::autograd::AutogradContext *ctx,
                            const torch::autograd::variable_list &grad_output) {

                        auto saved = ctx->get_saved_variables();
                        auto self = saved[0];
                        auto other = saved[1];

                        auto grads = detail::_izero_div_backward(
                                grad_output[0],
                                self,
                                other);
                        auto grad_self = std::get<0>(grads);
                        auto grad_other = std::get<1>(grads);

                        return {
                                grad_self,
                                grad_other,
                        };
                    }
                };

                class IZeroLeftDivideFunction
                        : public torch::autograd::Function<IZeroLeftDivideFunction> {
                public:
                    static torch::autograd::Variable forward(
                            torch::autograd::AutogradContext *ctx,
                            const torch::autograd::Variable &self,
                            const torch::autograd::Variable &other) {
                        at::AutoDispatchBelowADInplaceOrView g;
                        auto output = izero_ldiv(
                                self,
                                other);

                        ctx->save_for_backward({self, other});

                        return output;
                    }

                    static torch::autograd::variable_list backward(
                            torch::autograd::AutogradContext *ctx,
                            const torch::autograd::variable_list &grad_output) {

                        auto saved = ctx->get_saved_variables();
                        auto self = saved[0];
                        auto other = saved[1];

                        auto grads = detail::_izero_ldiv_backward(
                                grad_output[0],
                                self,
                                other);
                        auto grad_self = std::get<0>(grads);
                        auto grad_other = std::get<1>(grads);

                        return {
                                grad_self,
                                grad_other,
                        };
                    }
                };

                class IZeroDivideScalarFunction
                        : public torch::autograd::Function<IZeroDivideScalarFunction> {
                public:
                    static torch::autograd::Variable forward(
                            torch::autograd::AutogradContext *ctx,
                            const torch::autograd::Variable &self,
                            const at::Scalar &other) {
                        at::AutoDispatchBelowADInplaceOrView g;
                        auto output = izero_div(
                                self,
                                other);

                        ctx->save_for_backward({self});
                        ctx->saved_data["other"] = other;

                        return output;
                    }

                    static torch::autograd::variable_list backward(
                            torch::autograd::AutogradContext *ctx,
                            const torch::autograd::variable_list &grad_output) {

                        auto saved = ctx->get_saved_variables();
                        auto self = saved[0];
                        auto other = ctx->saved_data["other"].toScalar();

                        auto grad_self = detail::_izero_div_backward(
                                grad_output[0],
                                self,
                                other);

                        return {
                                grad_self,
                                torch::autograd::Variable(),
                        };
                    }
                };

                class IZeroLeftDivideScalarFunction
                        : public torch::autograd::Function<IZeroLeftDivideScalarFunction> {
                public:
                    static torch::autograd::Variable forward(
                            torch::autograd::AutogradContext *ctx,
                            const torch::autograd::Variable &self,
                            const at::Scalar &other) {
                        at::AutoDispatchBelowADInplaceOrView g;
                        auto output = izero_ldiv(
                                self,
                                other);

                        ctx->save_for_backward({self});
                        ctx->saved_data["other"] = other;

                        return output;
                    }

                    static torch::autograd::variable_list backward(
                            torch::autograd::AutogradContext *ctx,
                            const torch::autograd::variable_list &grad_output) {

                        auto saved = ctx->get_saved_variables();
                        auto self = saved[0];
                        auto other = ctx->saved_data["other"].toScalar();

                        auto grad_self = detail::_izero_ldiv_backward(
                                grad_output[0],
                                self,
                                other);

                        return {
                                grad_self,
                                torch::autograd::Variable(),
                        };
                    }
                };

                class IZeroDividerScalarFunction
                        : public torch::autograd::Function<IZeroDividerScalarFunction> {
                public:
                    static torch::autograd::Variable forward(
                            torch::autograd::AutogradContext *ctx,
                            const at::Scalar &self,
                            const torch::autograd::Variable &other) {
                        at::AutoDispatchBelowADInplaceOrView g;
                        auto output = izero_div(
                                self,
                                other);

                        ctx->save_for_backward({other});
                        ctx->saved_data["self"] = self;

                        return output;
                    }

                    static torch::autograd::variable_list backward(
                            torch::autograd::AutogradContext *ctx,
                            const torch::autograd::variable_list &grad_output) {

                        auto saved = ctx->get_saved_variables();
                        auto other = saved[0];
                        auto self = ctx->saved_data["self"].toScalar();

                        auto grad_other = detail::_izero_div_backward(
                                grad_output[0],
                                self,
                                other);

                        return {
                                torch::autograd::Variable(),
                                grad_other,
                        };
                    }
                };

                class IZeroLeftDividerScalarFunction
                        : public torch::autograd::Function<IZeroLeftDividerScalarFunction> {
                public:
                    static torch::autograd::Variable forward(
                            torch::autograd::AutogradContext *ctx,
                            const at::Scalar &self,
                            const torch::autograd::Variable &other) {
                        at::AutoDispatchBelowADInplaceOrView g;
                        auto output = izero_ldiv(
                                self,
                                other);

                        ctx->save_for_backward({other});
                        ctx->saved_data["self"] = self;

                        return output;
                    }

                    static torch::autograd::variable_list backward(
                            torch::autograd::AutogradContext *ctx,
                            const torch::autograd::variable_list &grad_output) {

                        auto saved = ctx->get_saved_variables();
                        auto other = saved[0];
                        auto self = ctx->saved_data["self"].toScalar();

                        auto grad_other = detail::_izero_ldiv_backward(
                                grad_output[0],
                                self,
                                other);

                        return {
                                torch::autograd::Variable(),
                                grad_other,
                        };
                    }
                };

                class IZeroDivideInplaceFunction
                        : public torch::autograd::Function<IZeroDivideInplaceFunction> {
                public:
                    static torch::autograd::Variable forward(
                            torch::autograd::AutogradContext *ctx,
                            torch::autograd::Variable &self,
                            const torch::autograd::Variable &other) {
                        at::AutoDispatchBelowADInplaceOrView g;
                        ctx->mark_dirty({self});
                        auto output = izero_div_(
                                self,
                                other);

                        ctx->save_for_backward({self, other});

                        return output;
                    }

                    static torch::autograd::variable_list backward(
                            torch::autograd::AutogradContext *ctx,
                            const torch::autograd::variable_list &grad_output) {

                        auto saved = ctx->get_saved_variables();
                        auto self = saved[0];
                        auto other = saved[1];

                        auto grads = detail::_izero_div__backward(
                                grad_output[0],
                                self,
                                other);
                        auto grad_self = std::get<0>(grads);
                        auto grad_other = std::get<1>(grads);

                        return {
                                grad_self,
                                grad_other,
                        };
                    }
                };

                class IZeroLeftDivideInplaceFunction
                        : public torch::autograd::Function<IZeroLeftDivideInplaceFunction> {
                public:
                    static torch::autograd::Variable forward(
                            torch::autograd::AutogradContext *ctx,
                            torch::autograd::Variable &self,
                            const torch::autograd::Variable &other) {
                        at::AutoDispatchBelowADInplaceOrView g;
                        ctx->mark_dirty({self});
                        auto output = izero_ldiv_(
                                self,
                                other);

                        ctx->save_for_backward({self, other});

                        return output;
                    }

                    static torch::autograd::variable_list backward(
                            torch::autograd::AutogradContext *ctx,
                            const torch::autograd::variable_list &grad_output) {

                        auto saved = ctx->get_saved_variables();
                        auto self = saved[0];
                        auto other = saved[1];

                        auto grads = detail::_izero_ldiv__backward(
                                grad_output[0],
                                self,
                                other);
                        auto grad_self = std::get<0>(grads);
                        auto grad_other = std::get<1>(grads);

                        return {
                                grad_self,
                                grad_other,
                        };
                    }
                };

                class IZeroDivideInplaceScalarFunction
                        : public torch::autograd::Function<IZeroDivideInplaceScalarFunction> {
                public:
                    static torch::autograd::Variable forward(
                            torch::autograd::AutogradContext *ctx,
                            torch::autograd::Variable &self,
                            const at::Scalar &other) {
                        at::AutoDispatchBelowADInplaceOrView g;
                        ctx->mark_dirty({self});
                        auto output = izero_div_(
                                self,
                                other);

                        ctx->save_for_backward({self});
                        ctx->saved_data["other"] = other;

                        return output;
                    }

                    static torch::autograd::variable_list backward(
                            torch::autograd::AutogradContext *ctx,
                            const torch::autograd::variable_list &grad_output) {

                        auto saved = ctx->get_saved_variables();
                        auto self = saved[0];
                        auto other = ctx->saved_data["other"].toScalar();

                        auto grad_self = detail::_izero_div__backward(
                                grad_output[0],
                                self,
                                other);

                        return {
                                grad_self,
                                torch::autograd::Variable(),
                        };
                    }
                };

                class IZeroLeftDivideInplaceScalarFunction
                        : public torch::autograd::Function<IZeroLeftDivideInplaceScalarFunction> {
                public:
                    static torch::autograd::Variable forward(
                            torch::autograd::AutogradContext *ctx,
                            torch::autograd::Variable &self,
                            const at::Scalar &other) {
                        at::AutoDispatchBelowADInplaceOrView g;
                        ctx->mark_dirty({self});
                        auto output = izero_ldiv_(
                                self,
                                other);

                        ctx->save_for_backward({self});
                        ctx->saved_data["other"] = other;

                        return output;
                    }

                    static torch::autograd::variable_list backward(
                            torch::autograd::AutogradContext *ctx,
                            const torch::autograd::variable_list &grad_output) {

                        auto saved = ctx->get_saved_variables();
                        auto self = saved[0];
                        auto other = ctx->saved_data["other"].toScalar();

                        auto grad_self = detail::_izero_ldiv__backward(
                                grad_output[0],
                                self,
                                other);

                        return {
                                grad_self,
                                torch::autograd::Variable(),
                        };
                    }
                };
            } // namespace

            at::Tensor izero_div_autograd(
                    const at::Tensor &self,
                    const at::Tensor &other) {
                return IZeroDivideFunction::apply(
                        self,
                        other);
            }

            at::Tensor izero_ldiv_autograd(
                    const at::Tensor &self,
                    const at::Tensor &other) {
                return IZeroLeftDivideFunction::apply(
                        self,
                        other);
            }

            at::Tensor izero_div_Scalar_autograd(
                    const at::Tensor &self,
                    const at::Scalar &other) {
                return IZeroDivideScalarFunction::apply(
                        self,
                        other);
            }

            at::Tensor izero_ldiv_Scalar_autograd(
                    const at::Tensor &self,
                    const at::Scalar &other) {
                return IZeroLeftDivideScalarFunction::apply(
                        self,
                        other);
            }

            at::Tensor izero_div_rScalar_autograd(
                    const at::Scalar &self,
                    const at::Tensor &other) {
                return IZeroDividerScalarFunction::apply(
                        self,
                        other);
            }

            at::Tensor izero_ldiv_rScalar_autograd(
                    const at::Scalar &self,
                    const at::Tensor &other) {
                return IZeroLeftDividerScalarFunction::apply(
                        self,
                        other);
            }

            at::Tensor izero_div__autograd(
                    at::Tensor &self,
                    const at::Tensor &other) {
                return IZeroDivideInplaceFunction::apply(
                        self,
                        other);
            }

            at::Tensor izero_ldiv__autograd(
                    at::Tensor &self,
                    const at::Tensor &other) {
                return IZeroLeftDivideInplaceFunction::apply(
                        self,
                        other);
            }

            at::Tensor izero_div__Scalar_autograd(
                    at::Tensor &self,
                    const at::Scalar &other) {
                return IZeroDivideInplaceScalarFunction::apply(
                        self,
                        other);
            }

            at::Tensor izero_ldiv__Scalar_autograd(
                    at::Tensor &self,
                    const at::Scalar &other) {
                return IZeroLeftDivideInplaceScalarFunction::apply(
                        self,
                        other);
            }

            TORCH_LIBRARY_IMPL(partialtorch, Autograd, m) {
                m.impl(
                        TORCH_SELECTIVE_NAME("partialtorch::_izero_div"),
                        TORCH_FN(izero_div_autograd));
                m.impl(
                        TORCH_SELECTIVE_NAME("partialtorch::_izero_ldiv"),
                        TORCH_FN(izero_ldiv_autograd));
                m.impl(
                        TORCH_SELECTIVE_NAME("partialtorch::_izero_div.Scalar"),
                        TORCH_FN(izero_div_Scalar_autograd));
                m.impl(
                        TORCH_SELECTIVE_NAME("partialtorch::_izero_ldiv.Scalar"),
                        TORCH_FN(izero_ldiv_Scalar_autograd));
                m.impl(
                        TORCH_SELECTIVE_NAME("partialtorch::_izero_div.rScalar"),
                        TORCH_FN(izero_div_rScalar_autograd));
                m.impl(
                        TORCH_SELECTIVE_NAME("partialtorch::_izero_ldiv.rScalar"),
                        TORCH_FN(izero_ldiv_rScalar_autograd));
                m.impl(
                        TORCH_SELECTIVE_NAME("partialtorch::_izero_div_"),
                        TORCH_FN(izero_div__autograd));
                m.impl(
                        TORCH_SELECTIVE_NAME("partialtorch::_izero_ldiv_"),
                        TORCH_FN(izero_ldiv__autograd));
                m.impl(
                        TORCH_SELECTIVE_NAME("partialtorch::_izero_div_.Scalar"),
                        TORCH_FN(izero_div__Scalar_autograd));
                m.impl(
                        TORCH_SELECTIVE_NAME("partialtorch::_izero_ldiv_.Scalar"),
                        TORCH_FN(izero_ldiv__Scalar_autograd));
            }
        }
    }
}
