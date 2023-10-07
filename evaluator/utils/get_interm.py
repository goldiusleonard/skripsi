import onnx
from onnx import helper

model = onnx.load("./weights/ssdg/ssdg_oulu_prot_1_simp_interm.onnx")
intermediate_layer_value_info = helper.ValueInfoProto()
intermediate_layer_value_info.name = "606"
model.graph.output.append(intermediate_layer_value_info)
onnx.save(model, "./weights/ssdg/ssdg_oulu_prot_1_simp_interm.onnx")
