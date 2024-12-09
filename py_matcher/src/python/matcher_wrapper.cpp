#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "template_matcher.h"

namespace py = pybind11;

// numpy array to cv::Mat conversion
cv::Mat numpy_uint8_to_mat(py::array_t<uint8_t>& array) {
    py::buffer_info buf = array.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Image must be 2-dimensional");
    }
    return cv::Mat(buf.shape[0], buf.shape[1], CV_8UC1, buf.ptr);
}

// Match result conversion
py::dict convert_match_result(const s_SingleTargetMatch& result) {
    py::dict dict;
    dict["score"] = result.dMatchScore;
    dict["angle"] = result.dMatchedAngle;
    dict["center_x"] = result.ptCenter.x;
    dict["center_y"] = result.ptCenter.y;
    dict["lt"] = py::make_tuple(result.ptLT.x, result.ptLT.y);
    dict["rt"] = py::make_tuple(result.ptRT.x, result.ptRT.y);
    dict["rb"] = py::make_tuple(result.ptRB.x, result.ptRB.y);
    dict["lb"] = py::make_tuple(result.ptLB.x, result.ptLB.y);
    return dict;
}

PYBIND11_MODULE(cvmatcher, m) {
    m.doc() = "Template matching module based on OpenCV"; 

    // Class definition
    py::class_<TemplateMatcher>(m, "TemplateMatcher")
        .def(py::init<>())
        // Main functions
        .def("set_source", [](TemplateMatcher& self, py::array_t<uint8_t> array) {
            cv::Mat mat = numpy_uint8_to_mat(array);
            self.SetSourceImage(mat);
        })
        .def("set_template", [](TemplateMatcher& self, py::array_t<uint8_t> array) {
            cv::Mat mat = numpy_uint8_to_mat(array);
            self.SetTemplateImage(mat);
            self.LearnPattern();
        })
        .def("match", [](TemplateMatcher& self) {
            if(self.Match()) {
                auto results = self.GetResults();
                py::list matches;
                for(const auto& result : results) {
                    matches.append(convert_match_result(result));
                }
                return matches;
            }
            return py::list();
        })
        // Parameter setters
        .def("set_max_positions", &TemplateMatcher::SetMaxPositions)
        .def("set_max_overlap", &TemplateMatcher::SetMaxOverlap)
        .def("set_score", &TemplateMatcher::SetScore)
        .def("set_tolerance_angle", &TemplateMatcher::SetToleranceAngle)
        .def("set_min_reduce_area", &TemplateMatcher::SetMinReduceArea)
        .def("set_debug_mode", &TemplateMatcher::SetDebugMode)
        .def("set_tolerance_range_mode", &TemplateMatcher::SetToleranceRangeMode)
        .def("set_tolerance_ranges", &TemplateMatcher::SetToleranceRanges)
        .def("set_stop_layer1", &TemplateMatcher::SetStopLayer1)
        .def("set_use_simd", &TemplateMatcher::SetUseSIMD)
        .def("set_use_subpixel", &TemplateMatcher::SetUseSubPixel);
}