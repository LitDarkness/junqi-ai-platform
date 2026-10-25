#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "neural_puct.h"
#include <cstring>
namespace py = pybind11;
using Tree = junqi::az::root_explore::NeuralPUCTTree;
py::dict leaf(const junqi::az::root_explore::PUCTLeafRequest& r) {
    py::dict d; d["terminal"]=r.terminal;d["terminal_value"]=r.terminal_value;
    d["state_key"]=r.state_key;d["position_key"]=r.position_key;d["player"]=r.player;
    d["legal_actions"]=r.legal_actions;d["repetition_contexts"]=r.repetition_contexts;
    if (!r.terminal) {
        py::array_t<float> f(std::vector<py::ssize_t>{junqi::az::AZState::kFeatureChannels,12,5});
        std::memcpy(f.mutable_data(),r.features.data(),r.features.size()*sizeof(float));d["features"]=std::move(f);
    } else d["features"]=py::none();
    return d;
}
PYBIND11_MODULE(JunqiAZRootExplore,m) {
    py::module_::import("JunqiAZ"); // Reuse the original AZState type registration; never duplicate it.
    py::class_<Tree>(m,"RootExploreTree")
      .def(py::init<const junqi::az::AZState&,float,bool,float>(),py::arg("root"),py::arg("cpuct")=1.5f,
           py::arg("guard")=false,py::arg("forced_coeff")=0.0f)
      .def("select_leaf",[](Tree& t){return leaf(t.select_leaf());})
      .def("select_leaves",[](Tree& t,int n){py::list a;for(auto& r:t.select_leaves(n))a.append(leaf(r));return a;})
      .def("backup",&Tree::backup).def("backup_leaves",&Tree::backup_leaves)
      .def("cancel_pending",&Tree::cancel_pending).def("advance_root",&Tree::advance_root)
      .def("root_key",&Tree::root_key).def("root_visits",&Tree::root_visits)
      .def("add_root_dirichlet",&Tree::add_root_dirichlet).def("has_pending_leaf",&Tree::has_pending_leaf)
      .def("root_tactical_stats",&Tree::root_tactical_stats)
      .def("forced_decisions",&Tree::forced_decisions).def("target_visits",&Tree::target_visits)
      .def("root_edges",[](const Tree& t){py::list a;for(auto& e:t.root_edges()){
          py::dict d;d["action"]=e.action;d["prior"]=e.prior;d["visits"]=e.visits;d["q"]=e.q;a.append(d);}return a;});
}
