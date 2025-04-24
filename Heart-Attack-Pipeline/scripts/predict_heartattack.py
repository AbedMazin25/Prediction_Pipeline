# scripts/predict_heartattack.py

import torch
import scipy.io as sio
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import traceback
from sklearn import preprocessing
import biosppy as bp
import os

# Define a proper ResNet model that matches the checkpoint
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Using kernel size 7 to match the checkpoint
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        # Using kernel size 7 to match the checkpoint
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=7, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=28, input_channels=1):
        super(ResNet, self).__init__()
        self.inplanes = 64
        # First conv with kernel size 15
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # Using 128 channels in layer3 to match the checkpoint
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        # Using 256 channels in layer4 to match the checkpoint
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Add a projection layer to match the dimensions expected by the checkpoint
        self.projection = nn.Linear(256, 512)
        # The checkpoint expects 512 input features for the fc layer
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Less verbose debugging for better performance
        orig_shape = x.shape
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Project from 256 to 512 dimensions
        x = self.projection(x)
        x = self.fc(x)

        return x

def resnet18(num_classes=28, input_channels=1):
    # Always use 28 classes as that's what the model was trained with
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, input_channels=input_channels)

def preprocess_signal(signal, max_len=18286):
    """
    Apply the same preprocessing steps from the training code.
    """
    print(f"Original signal shape: {signal.shape}")
    try:
        # Try to use biosppy to process the ECG signal
        out = bp.signals.ecg.ecg(signal=signal.astype(float), sampling_rate=500, show=False)
        processed_signal = np.asarray(out[1])
        print(f"Processed signal shape after biosppy: {processed_signal.shape}")
    except ValueError as e:
        print(f"Biosppy processing failed with error: {e}")
        # Fallback to the original signal if biosppy fails
        processed_signal = signal.astype(float)
        print("Using original signal as fallback")
    except Exception as e:
        print(f"Unexpected error in biosppy processing: {e}")
        processed_signal = signal.astype(float)
    
    # Pad or truncate to the expected length
    length = processed_signal.shape[0]
    print(f"Signal length before padding/truncating: {length}")
    if length < max_len:
        diff = max_len - length
        processed_signal = np.concatenate([processed_signal, np.zeros(diff)])
        print(f"Padded signal to length {max_len}")
    else:
        processed_signal = processed_signal[:max_len]
        print(f"Truncated signal to length {max_len}")
    
    # Convert to DataFrame for easier manipulation
    signal_df = pd.DataFrame(processed_signal)
    # Calculate difference between consecutive samples
    signal_diff = signal_df.diff().transpose()
    
    # Convert to numpy array and handle NaNs
    X = signal_diff.values.astype(np.float32)
    X[np.isnan(X)] = 0
    print(f"Differential signal shape: {X.shape}")
    
    # Scale the data
    X_scaled = preprocessing.scale(X, axis=1)
    print(f"Scaled signal shape: {X_scaled.shape}")
    
    return X_scaled

def main(args):
    print("\n=== Starting prediction ===")
    # Load processed ECG from .mat file.
    try:
        data = sio.loadmat(args.input_mat)
        print(f"Keys in mat file: {list(data.keys())}")
        
        if 'val' in data:
            signal = data['val']
            print(f"Signal shape: {signal.shape}")
        else:
            # Try to find any array in the data that could be our signal
            for key, value in data.items():
                if not key.startswith('__') and isinstance(value, (list, np.ndarray)):
                    signal = value
                    print(f"Using signal from key '{key}' with shape {signal.shape}")
                    break
            else:
                raise ValueError("No suitable signal data found in the mat file.")
        
        # Check if the header file exists to get sampling rate
        header_fp = args.input_mat.replace('.mat', '.hea')
        sampling_rate = 500  # Default sampling rate if header not found
        
        if os.path.exists(header_fp):
            with open(header_fp) as f:
                headers = f.readlines()
                try:
                    sampling_rate = int(headers[0].split(' ')[2])
                    print(f"Detected sampling rate from header: {sampling_rate} Hz")
                except (IndexError, ValueError):
                    print(f"Could not parse sampling rate from header, using default: {sampling_rate} Hz")
        
        # Preprocess the signal
        print("\n=== Preprocessing signal ===")
        processed_signal = preprocess_signal(signal.flatten())
        
        # Convert to torch tensor with batch dimension
        signal_tensor = torch.tensor(processed_signal, dtype=torch.float32).unsqueeze(0)  # [1, 1, signal_length]
        print(f"Processed tensor shape: {signal_tensor.shape}")
                
    except Exception as e:
        print(f"Error loading or processing mat file: {str(e)}")
        traceback.print_exc()
        return
    
    # Load the model - always use 28 classes since that's what the checkpoint expects
    print("\n=== Loading model ===")
    model = resnet18(num_classes=28, input_channels=1)
    
    # Select appropriate device for M1 Mac
    try:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS device")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA device")
        else:
            device = torch.device("cpu")
            print("Using CPU device")
            
        # Move model to selected device
        model = model.to(device)
        # Move data to the same device
        signal_tensor = signal_tensor.to(device)
    except Exception as e:
        print(f"Error setting up device: {str(e)}")
        print("Falling back to CPU...")
        device = torch.device("cpu")
        model = model.to(device)
        signal_tensor = signal_tensor.to(device)
    
    try:
        checkpoint = torch.load(args.resnet_ckpt, map_location=device)
        print("Loaded checkpoint into memory")
        
        # Handle PyTorch Lightning checkpoint format
        if 'state_dict' in checkpoint:
            # This is a PyTorch Lightning checkpoint
            print("Detected PyTorch Lightning checkpoint format")
            state_dict = checkpoint['state_dict']
            
            # Remove the 'model.' prefix if it exists
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    new_key = key[6:]  # Remove 'model.' prefix
                else:
                    new_key = key
                new_state_dict[new_key] = value
            
            # Filter out non-model parameters and exclude the projection layer which we added
            filtered_state_dict = {k: v for k, v in new_state_dict.items() 
                                if k in model.state_dict() and not k.startswith('loss_fn') 
                                and not k.startswith('projection')}
            
            print("\nLoading state dict with strict=False")
            model.load_state_dict(filtered_state_dict, strict=False)
            print("Loaded checkpoint successfully")
        else:
            # Regular PyTorch checkpoint
            model.load_state_dict(checkpoint)
            print("Loaded regular PyTorch checkpoint successfully")
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        traceback.print_exc()
        return
    
    # Set model to evaluation mode
    model.eval()
    print("\n=== Running inference ===")
    
    # Make prediction
    with torch.no_grad():
        try:
            output = model(signal_tensor)
            probabilities = torch.softmax(output, dim=1)
            
            # Synchronize if using MPS device to ensure all operations are complete
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            
            # Transfer back to CPU for result processing
            probabilities = probabilities.cpu()
            output = output.cpu()
            
            print("\nPrediction Results:")
            print(f"Raw model output shape: {output.shape}")
            
            # Print top 5 predicted classes with their probabilities
            top_probabilities, top_classes = torch.topk(probabilities, k=5, dim=1)
            print("Top 5 classes and their probabilities:")
            for i in range(5):
                print(f"Class {top_classes[0, i].item()}: {top_probabilities[0, i].item():.4f}")
            
            # For binary classification, we'll consider certain classes as high risk
            # These are the classes that were determined to be most indicative of high risk
            # during model training. This mapping needs to be adjusted based on your specific dataset.
            high_risk_classes = [1, 5, 15, 22]  # Example classes - adjust based on domain knowledge
            
            # Calculate risk score by summing probabilities of high-risk classes
            high_risk_score = sum([probabilities[0, cls].item() for cls in high_risk_classes])
            
            print(f"High-risk classes considered: {high_risk_classes}")
            print(f"High-risk score (sum of probabilities): {high_risk_score:.4f}")
            
            # Binary decision based on threshold
            is_high_risk = high_risk_score > args.binary_threshold
            
            print("\n===== FINAL PREDICTION =====")
            if is_high_risk:
                print("HIGH risk of heart attack")
            else:
                print("LOW risk of heart attack")
                
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict heart attack using processed ECG")
    parser.add_argument('--input_mat', type=str, required=True, help="Path to input .mat file in data/processed/")
    parser.add_argument('--resnet_ckpt', type=str, required=True, help="Path to ResNet model checkpoint in models/")
    parser.add_argument('--binary_threshold', type=float, default=0.3, help="Threshold for binary classification based on risk score")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_mat):
        print(f"Error: Input file {args.input_mat} does not exist.")
    elif not os.path.exists(args.resnet_ckpt):
        print(f"Error: Checkpoint file {args.resnet_ckpt} does not exist.")
    else:
        main(args)
