#pragma once

#include <c10/util/Exception.h>

namespace c10 {
    // Signal the end from iterator.__next__()
    class StopIteration : public Error {
        using Error::Error;
    };
}
