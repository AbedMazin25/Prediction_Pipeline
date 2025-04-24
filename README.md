# Heart Attack Pipeline System

This system integrates three components to provide real-time heart attack prediction from PCG (Phonocardiogram) signals:

1. **Heart Attack Pipeline** - Core ML pipeline that converts PCG to ECG and detects heart attacks
2. **Backend Server** - Django server that monitors for new files and runs the pipeline
3. **Frontend Dashboard** - React app that displays real-time pipeline results

## System Architecture

```
┌───────────────────┐     WebSocket     ┌──────────────────┐
│                   │    Connection     │                  │
│  React Frontend   │<------------------>│  Django Backend  │
│                   │                   │                  │
└───────────────────┘                   └──────────────────┘
                                               |  ^
                                               |  |
                                          File |  | Pipeline
                                        Watcher|  | Output
                                               |  |
                                               V  |
                                        ┌──────────────────┐
                                        │                  │
                                        │  Heart Attack    │
                                        │  Pipeline        │
                                        │                  │
                                        └──────────────────┘
```

## Getting Started

### 1. Setup Heart Attack Pipeline

```bash
cd ~/proj/Heart-Attack-Pipeline
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Make sure you have the necessary model weights in the `models/` directory:
- `enc_pixtopix_ecgppg_T7-V56.tm` - PCG to ECG conversion model
- `best-epoch=94-val_loss=0.01-val_f1=0.71.ckpt` - Heart attack prediction model

### 2. Setup Backend Server

```bash
cd ~/proj/backend-server/diagnose_server
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
```

### 3. Setup Frontend Dashboard

```bash
cd ~/proj/frontend/heart-web
npm install
# or if using yarn
yarn install
```

## Running the System

1. **Start the Backend Server**:

```bash
cd ~/proj/backend-server/diagnose_server
./start_server.sh
```

2. **Start the Frontend**:

```bash
cd ~/proj/frontend/heart-web
npm start
# or if using yarn
yarn start
```

3. **Test the System**:

Place a new `.npy` PCG file in `~/proj/Heart-Attack-Pipeline/data/` and the system will:
- Detect the new file
- Process it through the pipeline
- Stream results to the frontend in real-time
- Display heart attack predictions on the dashboard

## Component Documentation

Each component has its own detailed README with component-specific instructions:

- [Heart Attack Pipeline README](/Users/abedmatinpour/proj/Heart-Attack-Pipeline/README.md)
- [Backend Server README](/Users/abedmatinpour/proj/backend-server/diagnose_server/README.md)
- [Frontend Dashboard README](/Users/abedmatinpour/proj/frontend/heart-web/README.md)

## Troubleshooting

If you encounter issues with the system:

1. **WebSocket Connection Problems**:
   - Make sure the backend is running with Daphne (using start_server.sh)
   - Check browser console for connection errors
   - Verify the WebSocket URL in the frontend matches the backend

2. **Pipeline Not Running**:
   - Ensure paths in pipeline_watcher/consumers.py point to the correct locations
   - Verify model weights files exist
   - Check file permissions on the data directory

3. **Frontend Not Updating**:
   - Verify WebSocket connection is established
   - Check browser console for any errors
   - Ensure you're adding .npy files to the correct data directory 