#pragma once

#include <ATen/ATen.h>

namespace partialtorch {
    template<typename T>
    struct const_intrusive_ptr_arg {
        using type = const c10::intrusive_ptr<T> &;
    };

    template<typename T>
    using const_intrusive_ptr_arg_t = typename const_intrusive_ptr_arg<T>::type;

    // TODO: torch::Library::def doesn't support c10::intrusive_ptr_arg<T> & as
    //  function's arguments yet. This file must be removed when it does.
    template<typename T>
    struct intrusive_ptr_arg {
        using type = c10::intrusive_ptr<T>;
    };

    template<typename T>
    using intrusive_ptr_arg_t = typename intrusive_ptr_arg<T>::type;
}
