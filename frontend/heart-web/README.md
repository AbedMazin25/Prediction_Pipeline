# Heart Attack Dashboard Frontend

A React-based frontend application for the Heart Attack Pipeline system. This dashboard provides real-time visualization of ECG data and heart attack predictions from the pipeline.

## Features

- Real-time streaming of pipeline outputs via WebSockets
- Visual ECG signal display
- Heart attack detection alerts
- Vital signs monitoring (heart rate, blood pressure)
- User authentication system

## Setup

### Prerequisites

- Node.js 18+ and npm/yarn
- Backend server running with WebSocket support
- Heart Attack Pipeline configured properly

### Installation

1. Clone the repository
2. Install dependencies:

```bash
cd heart-web
npm install
# or if using yarn
yarn install
```

## Running the Application

Start the development server:

```bash
npm start
# or if using yarn
yarn start
```

This will launch the app at [http://localhost:3000](http://localhost:3000).

## WebSocket Connection

The frontend connects to the backend server's WebSocket endpoint (`ws://localhost:8000/ws/pipeline/`) to receive real-time updates from the Heart Attack Pipeline. The connection logic is implemented in the `PipelineOutput` component.

If your backend server is running on a different port or hostname, you'll need to update the WebSocket URL in `src/components/PipelineOutput.jsx`:

```javascript
const socketUrl = window.location.protocol === 'https:' 
  ? 'wss://' + window.location.host + '/ws/pipeline/' 
  : 'ws://' + (process.env.NODE_ENV === 'development' ? 'localhost:8000' : window.location.host) + '/ws/pipeline/';
```

## Building for Production

To create a production build:

```bash
npm run build
# or if using yarn
yarn build
```

This will generate optimized static files in the `build` folder, which can be served by any static web server.

## Integration with Backend

For the complete Heart Attack Pipeline system to work, make sure:

1. The Heart Attack Pipeline is set up correctly
2. The backend server is running with WebSocket support
3. The frontend is connected to the correct WebSocket endpoint

## Testing the System

To test the complete system:

1. Start the backend server using the start_server.sh script
2. Run the frontend app using `npm start`
3. Place a new .npy PCG file in the Heart Attack Pipeline's data directory
4. The backend will detect the new file, run the pipeline, and stream results to the frontend
5. The frontend will display the pipeline output and any detected heart attack warnings

## Troubleshooting

If you're not receiving real-time updates:

1. Check the browser console for WebSocket connection errors
2. Verify the backend server is running with Daphne for WebSocket support
3. Ensure the WebSocket URL in PipelineOutput.jsx matches your backend server address
4. Check that the Heart Attack Pipeline is properly configured and accessible to the backend
