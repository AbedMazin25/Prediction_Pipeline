# scripts/predict_ecg.py
# A script to predict heart attack risk from PCG signals by converting them to ECG first

import torch
import scipy.io as sio
import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing
import biosppy as bp
import os
import sys
import traceback


# Add the path to our main script so we can import from it
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import the model from the main script
from predict_heartattack import ResNet, BasicBlock, resnet18, preprocess_signal

# Import PCG to ECG conversion
sys.path.append(os.path.join(script_dir, '..'))
from pcg_to_ecg.conversion import convert_pcg_to_ecg

def predict_heart_attack_risk(pcg_file, model_checkpoint, threshold=0.3, fs=500):
    """
    Predict heart attack risk from a PCG .mat file by converting it to ECG first
    
    Parameters:
    -----------
    pcg_file : str
        Path to the .mat file containing PCG data
    model_checkpoint : str
        Path to the model checkpoint file
    threshold : float
        Threshold for binary classification (default: 0.3)
    fs : float
        Sampling frequency in Hz
        
    Returns:
    --------
    is_high_risk : bool
        True if high risk, False if low risk
    risk_score : float
        The calculated risk score
    top_classes : list
        List of top predicted class indices
    top_probs : list
        List of top class probabilities
    """
    try:
        # Load PCG data
        data = sio.loadmat(pcg_file)
        
        if 'val' in data:
            pcg_signal = data['val']
        else:
            # Try to find any array in the data that could be our signal
            for key, value in data.items():
                if not key.startswith('__') and isinstance(value, (list, np.ndarray)):
                    pcg_signal = value
                    break
            else:
                raise ValueError("No suitable signal data found in the mat file.")
        
        print(f"\nLoaded PCG signal with shape: {pcg_signal.shape}")
        
        # Convert PCG to ECG
        print("\nConverting PCG to ECG...")
        ecg_signal = convert_pcg_to_ecg(pcg_signal.flatten(), fs=fs)
        print(f"Converted ECG signal shape: {ecg_signal.shape}")
        
        # Save the converted ECG signal
        input_filename = os.path.basename(pcg_file)
        output_filename = f"converted_{input_filename}"
        output_path = os.path.join('data', 'processed', output_filename)
        
        # Create processed directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the converted ECG signal
        sio.savemat(output_path, {'val': ecg_signal})
        print(f"\nSaved converted ECG signal to: {output_path}")
        
        # Preprocess the converted ECG signal
        processed_signal = preprocess_signal(ecg_signal)
        
        # Convert to torch tensor
        signal_tensor = torch.tensor(processed_signal, dtype=torch.float32).unsqueeze(0)
        
        # Load the model
        model = resnet18(num_classes=28, input_channels=1)
        
        # Load the checkpoint
        checkpoint = torch.load(model_checkpoint, map_location='cpu')
        
        # Handle PyTorch Lightning checkpoint format
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            
            # Remove the 'model.' prefix if it exists
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    new_key = key[6:]  # Remove 'model.' prefix
                else:
                    new_key = key
                new_state_dict[new_key] = value
            
            # Filter out non-model parameters and exclude projection
            filtered_state_dict = {k: v for k, v in new_state_dict.items() 
                                 if k in model.state_dict() and not k.startswith('loss_fn')
                                 and not k.startswith('projection')}
            
            model.load_state_dict(filtered_state_dict, strict=False)
        else:
            # Regular PyTorch checkpoint
            model.load_state_dict(checkpoint)
        
        # Set model to evaluation mode
        model.eval()
        
        # Make prediction
        with torch.no_grad():
            output = model(signal_tensor)
            probabilities = torch.softmax(output, dim=1)
            
            # Get top 5 classes
            top_probs, top_classes = torch.topk(probabilities, k=5, dim=1)
            top_probs = top_probs.squeeze().tolist()
            top_classes = top_classes.squeeze().tolist()
            
            # Define high-risk classes (adjust based on your specific dataset)
            high_risk_classes = [1, 5, 15, 22]
            
            # Calculate risk score
            high_risk_score = sum([probabilities[0, cls].item() for cls in high_risk_classes])
            
            # Binary decision
            is_high_risk = high_risk_score > threshold
            
            return is_high_risk, high_risk_score, top_classes, top_probs
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        traceback.print_exc()
        return None, None, None, None

def main():
    parser = argparse.ArgumentParser(description="Predict heart attack risk from PCG data by converting to ECG first")
    parser.add_argument('--input', type=str, required=True, help="Path to input PCG .mat file")
    parser.add_argument('--model', type=str, default="models/best-epoch=94-val_loss=0.01-val_f1=0.71.ckpt", 
                      help="Path to model checkpoint")
    parser.add_argument('--threshold', type=float, default=0.3, 
                      help="Threshold for binary classification (default: 0.3)")
    parser.add_argument('--fs', type=float, default=500,
                      help="Sampling frequency in Hz (default: 500)")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        return
    
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} does not exist.")
        return
    
    print(f"\nAnalyzing PCG file: {args.input}")
    print(f"Using model: {args.model}")
    print(f"Classification threshold: {args.threshold}")
    print(f"Sampling frequency: {args.fs} Hz")
    
    is_high_risk, risk_score, top_classes, top_probs = predict_heart_attack_risk(
        args.input, args.model, args.threshold, args.fs)
    
    if is_high_risk is None:
        print("Failed to make a prediction. Check error messages above.")
        return
    
    print("\n===== Results =====")
    print(f"Heart attack risk score: {risk_score:.4f}")
    
    print("\nTop 5 predicted classes and their probabilities:")
    for i in range(5):
        print(f"Class {top_classes[i]}: {top_probs[i]:.4f}")
    
    print("\n===== FINAL PREDICTION =====")
    if is_high_risk:
        print("HIGH RISK OF HEART ATTACK")
    else:
        print("LOW RISK OF HEART ATTACK")

if __name__ == "__main__":
    main() 