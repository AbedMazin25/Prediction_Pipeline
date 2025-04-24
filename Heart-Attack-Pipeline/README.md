# ECG Heart Attack Risk Prediction

This project provides tools for predicting heart attack risk from ECG (electrocardiogram) data using a deep learning model.

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- SciPy
- scikit-learn
- pandas
- biosppy (for ECG signal processing)

To install all the required packages, run:

```bash
pip install torch numpy scipy scikit-learn pandas biosppy
```

## Quick Start


```bash
python scripts/run_pipeline.py pcg_to_ecg/testing_pcg/s_1000015.npy models/enc_pixtopix_ecgppg_T7-V56.tm models/best-epoch=94-val_loss=0.01-val_f1=0.71.ckpt
```

This script:
1. Loads the ECG data from a .mat file
2. Preprocesses the signal
3. Loads the pre-trained model
4. Makes a prediction about heart attack risk
5. Outputs the result (HIGH or LOW risk)

## Command-line Arguments

The `predict_ecg.py` script accepts the following arguments:

- `--input`: Path to the input .mat file containing ECG data (required)
- `--model`: Path to the model checkpoint file (default: models/best-epoch=94-val_loss=0.01-val_f1=0.71.ckpt)
- `--threshold`: Threshold for binary classification (default: 0.3)

Example with all arguments:

```bash
python scripts/predict_ecg.py --input data/processed/A0001.mat --model models/best-epoch=94-val_loss=0.01-val_f1=0.71.ckpt --threshold 0.35
```

## Data Format

The ECG data should be in a .mat file with either:
- A variable named 'val' containing the ECG signal
- Any other numerical array that can be used as an ECG signal

The script expects a multi-lead ECG signal but will flatten it for processing.

## Advanced Usage

For more advanced usage, you can use the `predict_heartattack.py` script which provides more detailed output and debugging information:

```bash
python scripts/predict_heartattack.py --input_mat path/to/your/ecg.mat --resnet_ckpt path/to/model/checkpoint.ckpt
```

## Model Information

The model used is a modified ResNet architecture trained on ECG data to predict heart attack risk. It was trained to classify ECGs into 28 different classes, which are then mapped to binary high/low risk categories.

The script uses a set of pre-determined high-risk classes (currently classes 1, 5, 15, and 22) to calculate an overall risk score. The binary classification is made by comparing this risk score against a threshold.

## About the Data

- The .mat files should contain ECG signals with a sampling rate around 500 Hz.
- If a header file (.hea) with the same base name exists in the same directory, the script will attempt to read the sampling rate from it. 

## Integration with Backend and Frontend

This Heart Attack Pipeline is designed to continuously process PCG files placed in the `data/` directory. When new `.npy` PCG files are added to the data directory, they are automatically processed by the pipeline.

### How it works

1. The backend server monitors the `data/` directory for new files
2. When a new `.npy` file is detected, the pipeline is automatically executed
3. Results are streamed in real-time to the frontend dashboard
4. The frontend displays pipeline output, ECG visualization, and heart attack predictions

### Requirements

- Python 3.9+
- PyTorch with GPU support (recommended)
- The model weights files in the `models/` directory

### Running the Pipeline Standalone

To run the pipeline manually:

```bash
# Activate the virtual environment
source venv/bin/activate

# Run the pipeline script
python scripts/run_pipeline.py <pcg_input_path> <conversion_weights> <prediction_weights>

# Example:
python scripts/run_pipeline.py data/sample.npy models/enc_pixtopix_ecgppg_T7-V56.tm models/best-epoch=94-val_loss=0.01-val_f1=0.71.ckpt
```

### Integration with Backend

The pipeline is automatically run by the backend server when new files are placed in the data directory. See the backend server README for more details on how to set up the full system.

For the integrated system to work properly, ensure this repository is located at:
```
/Users/abedmatinpour/proj/Heart-Attack-Pipeline
```

If you need to use a different location, update the paths in the backend server's pipeline_watcher/consumers.py file.

