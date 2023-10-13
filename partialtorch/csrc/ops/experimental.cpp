#include <iostream>

#include <ATen/ATen.h>
#include <torch/library.h>

#include "../MaskedPair.h"
#include "utils/mask_utils.h"
#include "utils/fill_identity.h"
#include "utils/schema_utils.h"

namespace partialtorch {
    namespace ops {
        namespace experimental {
            void test1() {
                std::cout << std::endl << "=== TEST ===" << std::endl;
                std::cout << utils::type_schema_str<std::vector<c10::intrusive_ptr<TensorMaskedPair>>>() << std::endl;
                std::cout << std::endl;
            }

            void test2() {
                std::cout << std::endl << "=== TEST ===" << std::endl;
                static constexpr auto op = utils::_ops::compose<
                        utils::_ops::cast<at::kDouble>,
                        utils::_ops::fill_identity_ones<false>>();
                auto x = at::rand({3, 3});
                auto x_mask = at::bernoulli(at::full_like(x, 0.5)).to(at::kBool);
                auto px = masked_pair(x, x_mask);
                auto pout = op.call(px);
                std::cout << pout << std::endl;
            }

            void test3() {
                std::cout << std::endl << "=== TEST ===" << std::endl;
                static const auto op = utils::_ops::sequential(
                        utils::_ops::cast<at::kDouble>(),
                        utils::_ops::fill_identity_state_value_(-0.5));
                auto x = at::rand({3, 3});
                auto x_mask = at::bernoulli(at::full_like(x, 0.5)).to(at::kBool);
                auto px = masked_pair(x, x_mask);
                auto pout = op.call(px);
                std::cout << pout << std::endl;
            }

            void test4() {
                std::cout << std::endl << "=== TEST ===" << std::endl;
                c10::optional<at::Tensor> m1 = at::bernoulli(at::full({4, 4}, 0.5)).to(at::kBool);
                c10::optional<at::Tensor> m2 = at::bernoulli(at::full({4, 4}, 0.5)).to(at::kBool);
                c10::optional<at::Tensor> m3 = at::bernoulli(at::full({4, 4}, 0.5)).to(at::kBool);
                c10::optional<at::Tensor> m4 = at::bernoulli(at::full({4, 4}, 0.5)).to(at::kBool);
                at::print(m1.value().bitwise_and(m2.value()).bitwise_and(m3.value()).bitwise_and(m4.value()));
                std::cout << std::endl;
                auto m = utils::all_masks(m1, m2, m3, m4);
                at::print(m.value());
            }

            TORCH_LIBRARY_FRAGMENT(partialtorch, m) {
                m.def("partialtorch::_test1", TORCH_FN(test1));
                m.def("partialtorch::_test2", TORCH_FN(test2));
                m.def("partialtorch::_test3", TORCH_FN(test3));
                m.def("partialtorch::_test4", TORCH_FN(test4));
            }
        }
    }
}
