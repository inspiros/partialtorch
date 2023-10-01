#pragma once

#include <torch/csrc/utils/pybind.h>

#include "../MaskedPair.h"
#include "../macros.h"

namespace pybind11 {
    namespace detail {
        template <>
        struct PARTIALTORCH_API type_caster<partialtorch::TensorMaskedPair> {
        public:
            // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
        PYBIND11_TYPE_CASTER(partialtorch::TensorMaskedPair, _("partialtorch.MaskedPair"));

            bool load(handle src, bool) {
                return true;
            }

            static handle cast(
                    const partialtorch::TensorMaskedPair& src,
                    return_value_policy /* policy */,
                    handle /* parent */);
        };
    }
}
