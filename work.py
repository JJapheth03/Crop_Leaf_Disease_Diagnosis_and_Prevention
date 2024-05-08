import torch
import torch.onnx

# Define your model class (replace MyModelClass with your actual model class)
class MyModelClass(torch.nn.Module):
    def __init__(self):
        super(MyModelClass, self).__init__()
        # Define your model architecture here

# Replace 'path_to_weights.pth' with the actual path to your saved model weights
weights_path = 'path_to_weights.pth'

# Instantiate the model
model = MyModelClass()

# Load the weights from a file (.pth usually)
state_dict = torch.load(weights_path)

# Load the weights into the model's architecture
model.load_state_dict(state_dict)

# Define the dummy input shape (adjust the values according to your model's input)
sample_batch_size = 1
channel = 3
height = 224
width = 224
dummy_input = torch.randn(sample_batch_size, channel, height, width)

# Export the model to ONNX format
onnx_path = "path_to_exported_model.onnx"
torch.onnx.export(model, dummy_input, onnx_path, verbose=True)
