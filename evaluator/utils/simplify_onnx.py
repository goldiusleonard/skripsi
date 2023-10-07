import onnx
from onnxsim import simplify

# load your predefined ONNX model
model = onnx.load("./weights/ssdg/ssdg_oulu_prot_1.onnx")

# convert model
model_simp, check = simplify(model)

assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, "./weights/ssdg_oulu_prot_1_simp.onnx")

# use model_simp as a standard ONNX model object