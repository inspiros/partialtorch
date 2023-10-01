#include <Python.h>

#include <torch/csrc/Exceptions.h>
#include <torch/script.h>

#include "partialtorch.h"
#include "PyMaskedPair.hpp"

#ifdef WITH_CUDA

#include <cuda.h>

#endif

namespace partialtorch {
    PyObject *initModule();
}

PyMODINIT_FUNC PyInit__C(void) {
    return partialtorch::initModule();
}

namespace partialtorch {
    PyObject *module;

    static std::vector<PyMethodDef> methods;

    static bool set_module_attr(const char *name, PyObject *v, bool incref = true) {
        // PyModule_AddObject steals reference
        if (incref)
            Py_INCREF(v);
        return PyModule_AddObject(module, name, v) == 0;
    }

    PyObject *initModule() {
        HANDLE_TH_ERRORS
                C10_LOG_API_USAGE_ONCE("partialtorch.python.import")

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ASSERT_TRUE(cmd) \
  if (!(cmd))            \
  return nullptr

                static struct PyModuleDef partialtorch_module = {
                        PyModuleDef_HEAD_INIT, "partialtorch._C", nullptr, -1, methods.data()};

                ASSERT_TRUE(module = PyModule_Create(&partialtorch_module));

                auto py_module = py::reinterpret_borrow<py::module>(module);

#ifdef WITH_CUDA
                PyObject *has_cuda = Py_True;
                PyObject *cuda_version = PyLong_FromLong(CUDA_VERSION);
#else
                PyObject *has_cuda = Py_False;
                    PyObject *cuda_version = PyLong_FromLong(-1);
#endif
                ASSERT_TRUE(set_module_attr("_has_cuda", has_cuda));
                ASSERT_TRUE(set_module_attr("_cuda_version", cuda_version));
                initPyMaskedPair(module);

                return module;
        END_HANDLE_TH_ERRORS
    }

    int64_t cuda_version() {
#ifdef WITH_CUDA
        return CUDA_VERSION;
#else
        return -1;
#endif
    }

    TORCH_LIBRARY_FRAGMENT(partialtorch, m) {
        m.def("_cuda_version", &cuda_version);
    }
}
