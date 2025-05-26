#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "lap.h"

namespace py = pybind11;

float wrapper_lap(
        py::array_t<float> cost_mat,
        py::array_t<int> rows,
        py::array_t<int> cols,
        py::array_t<float> u,
        py::array_t<float> v) {
    py::buffer_info cost_info = cost_mat.request();
    return lap<true, false, int, float>(
            cost_info.shape[0],
            static_cast<float*>(cost_info.ptr),
            static_cast<int*>(rows.request().ptr),
            static_cast<int*>(cols.request().ptr),
            static_cast<float*>(u.request().ptr),
            static_cast<float*>(v.request().ptr));
}


PYBIND11_MODULE(lap_uv, m) {
    m.doc() = "lap"; // optional module docstring
    m.def("lap", &wrapper_lap, "lap");
}
