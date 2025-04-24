#!/usr/bin/env python3
"""
Test script to simulate adding new files to the data directory to trigger processing.
"""

import os
import sys
import time
import shutil
import argparse
import numpy as np
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Add test files to the data directory')
    parser.add_argument('--file', help='Specific file to copy and modify')
    parser.add_argument('--count', type=int, default=1, help='Number of files to create')
    args = parser.parse_args()
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the data directory
    data_dir = os.path.join(script_dir, 'data')
    
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data directory: {data_dir}")
    
    # Create processed directory if it doesn't exist
    processed_dir = os.path.join(data_dir, 'processed')
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        print(f"Created processed directory: {processed_dir}")
    
    # Get source file to copy
    source_file = None
    if args.file and os.path.exists(args.file):
        source_file = args.file
    else:
        # Look for existing .npy files in the processed directory first
        for filename in os.listdir(processed_dir):
            if filename.endswith('.npy'):
                source_file = os.path.join(processed_dir, filename)
                break
                
        # If not found in processed, check main data directory
        if source_file is None:
            for filename in os.listdir(data_dir):
                if filename.endswith('.npy'):
                    source_file = os.path.join(data_dir, filename)
                    break
    
    if source_file is None:
        # If no existing file found, create a sample PCG file
        print("No existing .npy file found. Creating a sample PCG file.")
        source_file = os.path.join(data_dir, "sample_pcg.npy")
        # Create a simple synthetic PCG signal (random data)
        sample_data = np.random.randn(10000)
        np.save(source_file, sample_data)
    
    # Create and add new files
    for i in range(args.count):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"s_{1000000 + i}_{timestamp}.npy"
        new_file_path = os.path.join(data_dir, new_filename)
        
        # Copy the source file
        shutil.copy2(source_file, new_file_path)
        
        # Modify the copied file to make it slightly different
        data = np.load(new_file_path)
        modified_data = data + np.random.randn(*data.shape) * 0.1  # Add small random noise
        np.save(new_file_path, modified_data)
        
        print(f"Created new file: {new_file_path}")
        
        # Wait a bit to ensure file system events can be detected
        time.sleep(1)
    
    print(f"Added {args.count} new file(s) to the data directory.")
    print("The pipeline should detect and process these files.")

if __name__ == "__main__":
    main() 