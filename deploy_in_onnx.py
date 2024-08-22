import time
from tqdm import tqdm
import torch
import onnx
import onnxruntime as ort

import torch.onnx.symbolic_helper as sym_help
import torch.onnx.symbolic_opset11 as opset11
from torch.onnx.symbolic_helper import parse_args, _unimplemented

from utils import convert_state_dict
from models import restormer_arch


@parse_args('v', 'i')
def pixel_unshuffle(g, self, downscale_factor):
    rank = sym_help._get_tensor_rank(self)
    if rank is not None and rank != 4:
        return _unimplemented('pixel_unshuffler', msg='Only 4D tensor input is supported.')
    return g.op('SpaceToDepth', self, blocksize_i=downscale_factor)

opset11.pixel_unshuffle = pixel_unshuffle


# Step 1: Assuming you've already set up the environment
model_path = 'checkpoints/docres.pkl'
device_type = 'cpu'
onnx_model_path = 'checkpoints/docres.onnx'

# Step 2: Load the model (you'll need to define the model architecture)
model = restormer_arch.Restormer(
    inp_channels=6,
    out_channels=3,
    dim=48,
    num_blocks=[2, 3, 3, 4],
    num_refinement_blocks=4,
    heads=[1, 2, 4, 8],
    ffn_expansion_factor=2.66,
    bias=False,
    LayerNorm_type='WithBias',
    dual_pixel_task=True
)

if device_type == 'cpu':
    state = convert_state_dict(torch.load(
        model_path, map_location='cpu')['model_state'])
else:
    state = convert_state_dict(torch.load(
        model_path, map_location='cuda:0')['model_state'])
model.load_state_dict(state)
model.eval()

# Step 3: Prepare dummy input
dummy_input = torch.rand(1, 6, 512, 512)

# Step 4: Export to ONNX
torch.onnx.export(model, dummy_input, onnx_model_path,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}},
                  export_params=True,
                  opset_version=11)

# Step 5: Verify the ONNX model
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
print("Model successfully converted and verified!")

# Test with ONNX Runtime
ort_session = ort.InferenceSession(onnx_model_path)
ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}

# Measure inference time
num_iterations = 10
total_time = 0
for _ in tqdm(range(num_iterations)):
    start_time = time.time()
    ort_outputs = ort_session.run(None, ort_inputs)
    end_time = time.time()
    total_time += (end_time - start_time)

average_time = total_time / num_iterations
print(f"Average inference time over {num_iterations} iterations: {average_time:.4f} seconds")
