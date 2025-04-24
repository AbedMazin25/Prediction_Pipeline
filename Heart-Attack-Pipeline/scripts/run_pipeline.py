import subprocess
import os
import sys
import time
import matplotlib.pyplot as plt
import platform
import numpy as np

def run_pipeline(pcg_input_path, conversion_weights, prediction_weights, output_dir="pcg_to_ecg", debug=False, measure_latency=False):
    """
    Run the complete pipeline: PCG to ECG conversion followed by heart attack prediction
    
    Parameters:
    -----------
    pcg_input_path : str
        Path to the input PCG file
    conversion_weights : str
        Path to the PCG to ECG conversion model weights
    prediction_weights : str
        Path to the heart attack prediction model weights
    output_dir : str
        Directory to save intermediate and final results
    debug : bool
        Whether to show debug plots
    measure_latency : bool
        Whether to measure inference latency
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine the correct venv activation command based on OS
    is_windows = platform.system() == "Windows"
    if is_windows:
        venv_activate = "venv\\Scripts\\activate.bat"
    else:
        venv_activate = "source venv/bin/activate"
    
    if measure_latency:
        print("\n=== Measuring Full Pipeline Latency (10 runs) ===")
        latency_times = []
        
        for i in range(10):
            print(f"\nRun {i+1}/10:")
            start_time = time.time()
            
            # Step 1: Convert PCG to ECG (silent mode for latency measurement)
            print(f"  Converting PCG to ECG...")
            output_ecg = os.path.join(output_dir, f"run_{i}_pred_ecg.mat")
            
            if is_windows:
                conversion_cmd = f'cmd /c "{venv_activate} && python pcg_to_ecg/conversion.py --input {pcg_input_path} --weights {conversion_weights} --output {output_ecg} --fs 1000 --smooth_sec 0.02 --polyorder 1 --no-plot"'
            else:
                conversion_cmd = f'{venv_activate} && python pcg_to_ecg/conversion.py --input {pcg_input_path} --weights {conversion_weights} --output {output_ecg} --fs 1000 --smooth_sec 0.02 --polyorder 1 --no-plot'
            
            # Run conversion with suppressed output
            subprocess.run(conversion_cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Step 2: Run prediction (silent mode)
            print(f"  Running prediction...")
            
            if is_windows:
                prediction_cmd = f'cmd /c "{venv_activate} && python scripts/predict_heartattack.py --input_mat {output_ecg} --resnet_ckpt {prediction_weights}"'
            else:
                prediction_cmd = f'{venv_activate} && python scripts/predict_heartattack.py --input_mat {output_ecg} --resnet_ckpt {prediction_weights}'
            
            # Run prediction with suppressed output and error handling
            try:
                process = subprocess.Popen(prediction_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                process.wait()
                if process.returncode != 0:
                    print(f"  Warning: Prediction command exited with non-zero code: {process.returncode}")
            except Exception as e:
                print(f"  Error running prediction: {str(e)}")
            
            end_time = time.time()
            run_time = end_time - start_time
            latency_times.append(run_time)
            print(f"  Total time: {run_time:.4f} seconds")
            
            # Clean up the temporary file
            if os.path.exists(output_ecg):
                os.remove(output_ecg)
        
        # Calculate statistics
        mean_latency = np.mean(latency_times)
        std_dev = np.std(latency_times)
        min_latency = np.min(latency_times)
        max_latency = np.max(latency_times)
        
        print("\n=== Latency Measurement Results ===")
        print(f"Mean latency: {mean_latency:.4f} seconds ({mean_latency*1000:.2f} ms)")
        print(f"Standard deviation: {std_dev:.4f} seconds ({std_dev*1000:.2f} ms)")
        print(f"Min latency: {min_latency:.4f} seconds ({min_latency*1000:.2f} ms)")
        print(f"Max latency: {max_latency:.4f} seconds ({max_latency*1000:.2f} ms)")
        
        print("\n=== Running final pipeline with output ===")
    
    # Regular pipeline execution with output
    # Step 1: Convert PCG to ECG
    print("\n=== Converting PCG to ECG ===")
    output_ecg = os.path.join(output_dir, "pred_ecg.mat")
    
    # Add a --no-plot flag if not in debug mode
    plot_flag = "" if debug else "--no-plot"
    
    # Construct a command that activates venv first, then runs the conversion script
    if is_windows:
        conversion_cmd = f'cmd /c "{venv_activate} && python pcg_to_ecg/conversion.py --input {pcg_input_path} --weights {conversion_weights} --output {output_ecg} --fs 1000 --smooth_sec 0.02 --polyorder 1 {plot_flag}"'
    else:
        conversion_cmd = f'{venv_activate} && python pcg_to_ecg/conversion.py --input {pcg_input_path} --weights {conversion_weights} --output {output_ecg} --fs 1000 --smooth_sec 0.02 --polyorder 1 {plot_flag}'
    
    print(f"Running conversion: {conversion_cmd}")
    process = subprocess.Popen(conversion_cmd, shell=True)
    
    # Wait for the process to complete
    process.wait()
    
    # Wait for a moment to ensure the file is written
    time.sleep(2)
    
    # Check if the output file exists
    if not os.path.exists(output_ecg):
        print(f"Error: Output ECG file {output_ecg} was not created!")
        return
    
    # Step 2: Run heart attack prediction
    print("\n=== Running Heart Attack Prediction ===")
    
    # Construct a command that activates venv first, then runs the prediction script
    if is_windows:
        prediction_cmd = f'cmd /c "{venv_activate} && python scripts/predict_heartattack.py --input_mat {output_ecg} --resnet_ckpt {prediction_weights}"'
    else:
        prediction_cmd = f'{venv_activate} && python scripts/predict_heartattack.py --input_mat {output_ecg} --resnet_ckpt {prediction_weights}'
    
    print(f"Running prediction: {prediction_cmd}")
    try:
        process = subprocess.Popen(prediction_cmd, shell=True)
        process.wait()
        if process.returncode != 0:
            print(f"Warning: Prediction command exited with non-zero code: {process.returncode}")
            print("Continuing with pipeline...")
    except Exception as e:
        print(f"Error running prediction: {str(e)}")
        print("Continuing with pipeline...")

if __name__ == "__main__":
    # Check for flags
    debug_mode = False
    measure_latency = False
    
    # Process command-line arguments
    args_to_process = sys.argv.copy()
    for arg in args_to_process:
        if arg.lower() == 'debug':
            debug_mode = True
            sys.argv.remove(arg)
        elif arg.lower() == '--measure-latency':
            measure_latency = True
            sys.argv.remove(arg)
    
    if len(sys.argv) != 4:
        print("Usage: python run_pipeline.py <pcg_input_path> <conversion_weights> <prediction_weights> [debug] [--measure-latency]")
        print("Example: python run_pipeline.py pcg_to_ecg/testing_pcg/s_1000015.npy models/enc_pixtopix_ecgppg_T7-V56.tm models/best-epoch=94-val_loss=0.01-val_f1=0.71.ckpt --measure-latency")
        sys.exit(1)
    
    pcg_input = sys.argv[1]
    conv_weights = sys.argv[2]
    pred_weights = sys.argv[3]
    
    run_pipeline(pcg_input, conv_weights, pred_weights, debug=debug_mode, measure_latency=measure_latency) 