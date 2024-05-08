import torch
import torch.nn as nn

# Define the SimpleCNN class
class SimpleCNN(nn.Module):
     def __init__(self, num_classes=39):
        super(SimpleCNN, self).__init__()
        
        # Define the convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Define additional convolutional layers (repeat the process)
        self.conv_layers_additional = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2)
            # Add more convolutional layers if needed
        )
        
        # Define the fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 56 * 56, 128),  # Adjust input size based on output shape after convolutions
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

def forward(self, x):
        # Pass input through the first set of convolutional layers
        x = self.conv_layers(x)
        
        # Pass input through additional convolutional layers
        x = self.conv_layers_additional(x)
        
        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Pass input through fully connected layers
        x = self.fc_layers(x)
        
        return x

# Define MyModel with specific conv_layers and use fc_layers from SimpleCNN
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2)
        )

        # Load the state dictionary directly and extract required keys
        state_dict = torch.load("C:\\Users\\91990\\OneDrive\\Documents\\Plant-Disease-Detection-main\\Flask Deployed App\\plant_disease_model_1_latest.pt")
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 112 * 112, 128),  # Adjust input size based on output shape after convolutions
            nn.ReLU(),
            nn.Linear(128, 39)  # Assuming 39 is the number of output classes
        )
        # Update the state dictionary keys for fc_layers
        self.fc_layers.load_state_dict(state_dict, strict=False)  # Loading only the required keys

# Instantiate MyModel
my_model = MyModel()

# Save the weights of MyModel as converted_model.pth
torch.save(my_model.state_dict(), 'converted_model.pth')
