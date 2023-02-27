import onnx
from onnx.tools import update_model_dims


def _onnx_rename(model, names, names_new):
  for node in model.graph.node:
    for i, n in enumerate(node.input):
      if n in names:
        node.input[i] = names_new[names.index(n)]
    for i, n in enumerate(node.output):
      if n in names:
        node.output[i] = names_new[names.index(n)]
  for node in model.graph.input:
    if node.name in names:
      node.name = names_new[names.index(node.name)]
  # print(model.graph.input)
  for node in model.graph.output:
    if node.name in names:
      node.name = names_new[names.index(node.name)]
  # print(model.graph.output)

model = onnx.load("model.onnx")

# Rename input and output names
_onnx_rename(model, ["input_1", "conv2d_6"], ["input", "output"])

# Change batch size
variable_length_model = update_model_dims.update_inputs_outputs_dims(model, {"input": [1, 224, 224, 3]}, {"output": [1, 224, 224, 3]})

onnx.save(model, 'model.onnx')