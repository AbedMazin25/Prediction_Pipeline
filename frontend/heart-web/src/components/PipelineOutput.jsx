import React, { useEffect, useState, useRef } from 'react';

// Create stylesheet for the component
const styles = {
  pipelineOutput: {
    marginTop: '20px',
    padding: '15px',
    borderRadius: '8px',
    backgroundColor: '#f8f9fa',
    boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)'
  },
  connectionStatus: {
    display: 'flex',
    alignItems: 'center',
    marginBottom: '10px',
    fontWeight: 500
  },
  statusIndicator: {
    display: 'inline-block',
    width: '10px',
    height: '10px',
    borderRadius: '50%',
    marginRight: '8px'
  },
  connected: {
    backgroundColor: '#28a745'
  },
  disconnected: {
    backgroundColor: '#dc3545'
  },
  pipelineStatus: {
    marginBottom: '15px',
    fontSize: '16px'
  },
  statusProcessing: {
    color: '#fd7e14',
    fontWeight: 'bold'
  },
  statusComplete: {
    color: '#28a745',
    fontWeight: 'bold'
  },
  statusError: {
    color: '#dc3545',
    fontWeight: 'bold'
  },
  heartAttackAlert: {
    marginTop: '10px',
    padding: '10px',
    backgroundColor: '#dc3545',
    color: 'white',
    fontWeight: 'bold',
    textAlign: 'center',
    borderRadius: '4px',
    animation: 'pulse 2s infinite'
  },
  outputConsole: {
    marginTop: '20px'
  },
  messageContainer: {
    height: '300px',
    overflowY: 'auto',
    padding: '10px',
    border: '1px solid #ced4da',
    borderRadius: '4px',
    backgroundColor: '#212529',
    color: '#f8f9fa',
    fontFamily: 'monospace'
  },
  message: {
    marginBottom: '5px',
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word'
  },
  errorMessage: {
    color: '#ff6b6b'
  },
  outputMessage: {
    color: '#ced4da'
  },
  completeMessage: {
    color: '#69db7c'
  },
  highlightMessage: {
    color: '#ffd43b',
    fontWeight: 'bold',
    fontSize: '1.1em',
    padding: '2px 0'
  },
  diagnosisMessage: {
    color: '#4dabf7',
    fontWeight: 'bold',
    fontSize: '1.2em',
    padding: '5px',
    margin: '5px 0',
    borderRadius: '4px',
    backgroundColor: 'rgba(0, 123, 255, 0.1)'
  },
  diagnosisLow: {
    backgroundColor: 'rgba(40, 167, 69, 0.2)',
    color: '#2ecc71'
  },
  diagnosisMedium: {
    backgroundColor: 'rgba(255, 193, 7, 0.2)',
    color: '#f39c12'
  },
  diagnosisHigh: {
    backgroundColor: 'rgba(220, 53, 69, 0.2)',
    color: '#e74c3c'
  },
  vitalsContainer: {
    display: 'flex',
    gap: '20px',
    marginBottom: '15px'
  },
  vitalItem: {
    flex: 1,
    padding: '10px',
    borderRadius: '8px',
    backgroundColor: 'white',
    boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)'
  },
  vitalLabel: {
    fontSize: '14px',
    color: '#6c757d',
    marginBottom: '2px'
  },
  vitalValue: {
    fontSize: '18px',
    fontWeight: 600,
    color: '#212529'
  },
  diagnosisResult: {
    padding: '35px 30px',
    borderRadius: '14px',
    marginTop: '10px',
    marginBottom: '15px',
    boxShadow: '0 8px 20px rgba(0, 0, 0, 0.08)',
    textAlign: 'center',
    transition: 'all 0.5s ease',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    animation: 'fadeIn 0.8s ease-in-out',
    width: '100%',
    background: 'white',
  },
  diagnosisTitle: {
    fontSize: '28px',
    fontWeight: '800',
    marginBottom: '20px',
    textTransform: 'uppercase',
    letterSpacing: '1.5px',
    background: 'linear-gradient(45deg, #333, #777)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
  },
  diagnosisIcon: {
    fontSize: '72px',
    marginBottom: '20px',
    animation: 'pulseScale 2s infinite',
  },
  diagnosisDescription: {
    fontSize: '18px',
    marginTop: '15px',
    maxWidth: '90%',
    lineHeight: '1.6',
  },
  diagnosisResultLow: {
    background: 'linear-gradient(135deg, rgba(40, 167, 69, 0.05), rgba(40, 167, 69, 0.12))',
    borderLeft: '8px solid #28a745',
    color: '#155724',
  },
  diagnosisResultMedium: {
    background: 'linear-gradient(135deg, rgba(255, 193, 7, 0.05), rgba(255, 193, 7, 0.12))',
    borderLeft: '8px solid #ffc107',
    color: '#856404',
  },
  diagnosisResultHigh: {
    background: 'linear-gradient(135deg, rgba(220, 53, 69, 0.05), rgba(220, 53, 69, 0.12))',
    borderLeft: '8px solid #dc3545',
    color: '#721c24',
  },
  diagnosisResultUnknown: {
    background: 'linear-gradient(135deg, rgba(108, 117, 125, 0.05), rgba(108, 117, 125, 0.12))',
    borderLeft: '8px solid #6c757d',
    color: '#343a40',
  },
  waitingContainer: {
    padding: '40px 30px',
    textAlign: 'center',
    borderRadius: '16px',
    marginTop: '20px',
    marginBottom: '20px',
    boxShadow: '0 6px 18px rgba(0, 0, 0, 0.08)',
    background: 'linear-gradient(to bottom, #f8f9fa, #e9ecef)',
    maxWidth: '800px',
    margin: '30px auto',
  },
  waitingIcon: {
    fontSize: '48px',
    marginBottom: '20px',
    animation: 'spin 3s infinite linear',
  },
  waitingTitle: {
    fontSize: '24px',
    fontWeight: '600',
    color: '#495057',
    marginBottom: '15px',
  },
  waitingDescription: {
    fontSize: '16px',
    color: '#6c757d',
  },
  toggleButton: {
    padding: '10px 18px',
    borderRadius: '8px',
    border: 'none',
    background: 'linear-gradient(135deg, #f5f5f5, #e5e5e5)',
    color: '#444',
    fontSize: '14px',
    fontWeight: '500',
    cursor: 'pointer',
    marginTop: '15px',
    transition: 'all 0.2s ease',
    boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)',
    letterSpacing: '0.3px',
  },
  monitoringAlert: {
    fontSize: '14px',
    color: '#6c757d',
    backgroundColor: 'rgba(108, 117, 125, 0.05)',
    padding: '10px 15px',
    borderRadius: '8px',
    marginTop: '15px',
    borderLeft: '3px solid #6c757d',
    display: 'flex',
    alignItems: 'center',
    maxWidth: '90%',
  },
  pulsingDot: {
    width: '8px',
    height: '8px',
    backgroundColor: '#4dabf7',
    borderRadius: '50%',
    marginRight: '10px',
    animation: 'pulseScale 1.5s ease-in-out infinite',
  },
  mainContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: '25px',
    maxWidth: '1000px',
    margin: '0 auto',
  },
  cardContainer: {
    background: 'linear-gradient(to bottom, #ffffff, #f9f9f9)',
    borderRadius: '16px',
    boxShadow: '0 8px 24px rgba(0, 0, 0, 0.1)',
    padding: '28px',
    transition: 'all 0.3s ease',
    border: '1px solid rgba(230, 230, 230, 0.7)',
  },
  cardTitle: {
    fontSize: '22px',
    fontWeight: '700',
    marginBottom: '18px',
    color: '#333',
    borderBottom: '1px solid #e9ecef',
    paddingBottom: '12px',
    letterSpacing: '0.3px',
  },
  spinnerContainer: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    padding: '20px 0',
  },
  spinner: {
    width: '60px',
    height: '60px',
    border: '4px solid rgba(0, 0, 0, 0.05)',
    borderRadius: '50%',
    borderTop: '4px solid #4285F4',
    animation: 'spin 1s linear infinite',
  },
  processingIndicator: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    marginTop: '15px',
  },
  processingLabel: {
    fontSize: '16px',
    fontWeight: '500',
    color: '#4285F4',
    marginTop: '15px',
    letterSpacing: '0.3px',
  },
  noResultsMessage: {
    textAlign: 'center',
    padding: '40px 20px',
    color: '#5f6368',
    fontSize: '20px',
    fontWeight: '500',
    background: 'linear-gradient(135deg, #f8f9fa, #f1f3f4)',
    borderRadius: '12px',
    boxShadow: 'inset 0 1px 3px rgba(0, 0, 0, 0.05)',
  },
};

// Add keyframes for all animations
const keyframesAnimation = `
@keyframes pulse {
  0% { opacity: 1; }
  50% { opacity: 0.5; }
  100% { opacity: 1; }
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(15px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes pulseScale {
  0% { transform: scale(1); }
  50% { transform: scale(1.1); }
  100% { transform: scale(1); }
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

@keyframes fadeInRight {
  from { opacity: 0; transform: translateX(-15px); }
  to { opacity: 1; transform: translateX(0); }
}

@keyframes breathe {
  0% { transform: scale(1); opacity: 0.8; }
  50% { transform: scale(1.05); opacity: 1; }
  100% { transform: scale(1); opacity: 0.8; }
}
`;

function PipelineOutput() {
  const [connected, setConnected] = useState(false);
  const [messages, setMessages] = useState([]);
  const [status, setStatus] = useState('idle');
  const [heartAttackDetected, setHeartAttackDetected] = useState(false);
  const [heartRate, setHeartRate] = useState(null);
  const [bloodPressure, setBloodPressure] = useState({ systolic: null, diastolic: null });
  const [diagnosisResult, setDiagnosisResult] = useState(null);
  const ws = useRef(null);
  const messagesEndRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const pollIntervalRef = useRef(null);
  const [showConsole, setShowConsole] = useState(false);
  
  // For debugging - add a test message function
  const addTestMessage = (message, status = 'output') => {
    console.log('Adding test message:', message);
    setMessages(prev => [...prev, { message, status }]);
  };

  // Add pulse animation to document head
  useEffect(() => {
    const styleEl = document.createElement('style');
    styleEl.innerHTML = keyframesAnimation;
    document.head.appendChild(styleEl);
    return () => document.head.removeChild(styleEl);
  }, []);

  // Auto-scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Function to trigger a poll for the latest diagnosis
  const pollForResults = () => {
    console.log('Polling for latest results...');
    if (ws.current?.readyState === WebSocket.OPEN) {
      try {
        ws.current.send(JSON.stringify({
          action: 'poll_latest_results'
        }));
        console.log('Sent poll request');
      } catch (error) {
        console.error('Error sending poll request:', error);
      }
    } else {
      console.log('WebSocket not open, cannot poll for results');
    }
  };

  // Start automatic polling when connected
  useEffect(() => {
    if (connected) {
      // Clear any existing poll interval
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
      
      // Set up polling every 5 seconds
      pollIntervalRef.current = setInterval(() => {
        pollForResults();
      }, 5000);
      
      // Initial poll immediately after connection
      pollForResults();
      
      // Return cleanup function
      return () => {
        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
        }
      };
    }
  }, [connected]);

  // Handle a received message
  const handleMessage = (data) => {
    if (data.type === 'pipeline_update') {
      console.log('Processing pipeline update:', data.data);
      
      // Add the new message to our messages array
      setMessages(prev => [...prev, data.data]);
      
      // Update status based on the message
      if (data.data.status) {
        setStatus(data.data.status);
        
        // Process diagnosis messages specially
        if (data.data.status === 'diagnosis') {
          console.log('Received diagnosis data:', data.data);
          // Use function form to ensure we're working with latest state
          setDiagnosisResult({
            message: data.data.message,
            riskLevel: data.data.risk_level || 'unknown'
          });
          
          // If high risk, set heart attack detected
          if (data.data.risk_level === 'high') {
            setHeartAttackDetected(true);
          }
        }
        
        // When a new processing starts, clear previous diagnosis results
        // This ensures we don't show old results for new files
        if (data.data.status === 'processing' && data.data.message && 
            data.data.message.includes('Processing file:')) {
          console.log('New file processing started, clearing previous results');
          setDiagnosisResult(null);
          setHeartAttackDetected(false);
          setHeartRate(null);
          setBloodPressure({ systolic: null, diastolic: null });
        }
      }
      
      // Look for heart attack detection in the message
      if (data.data.message && typeof data.data.message === 'string') {
        const message = data.data.message.toLowerCase();
        
        // Check for heart attack indication in the message
        if (message.includes('heart attack') || 
            message.includes('positive for myocardial infarction') ||
            message.includes('high risk')) {
          setHeartAttackDetected(true);
        }
        
        // Look for heart rate information
        const heartRateMatch = data.data.message.match(/heart rate:\s*(\d+)/i);
        if (heartRateMatch && heartRateMatch[1]) {
          setHeartRate(parseInt(heartRateMatch[1], 10));
        }
        
        // Look for blood pressure information
        const bpMatch = data.data.message.match(/blood pressure:\s*(\d+)\/(\d+)/i);
        if (bpMatch && bpMatch[1] && bpMatch[2]) {
          setBloodPressure({
            systolic: parseInt(bpMatch[1], 10),
            diastolic: parseInt(bpMatch[2], 10)
          });
        }
        
        // Also check for prediction directly in the output
        if (message.includes('final prediction') || message.includes('risk of heart attack')) {
          const riskLevel = 
            message.includes('high risk') ? 'high' :
            message.includes('medium risk') || message.includes('moderate risk') ? 'medium' :
            message.includes('low risk') ? 'low' : 'unknown';
            
          setDiagnosisResult({
            message: `DIAGNOSIS: ${message.toUpperCase()}`,
            riskLevel: riskLevel
          });
        }
      }
    } else if (data.type === 'connection_established') {
      console.log('Connection established message:', data.message);
      addTestMessage('Connection established with server', 'info');
    } else if (data.type === 'status') {
      console.log('Status message:', data.message);
      addTestMessage(`Status: ${data.message}`, 'info');
    } else {
      console.log('Unknown message type:', data.type, data);
      addTestMessage(`Received message: ${JSON.stringify(data)}`, 'info');
    }
  };

  useEffect(() => {
    // Add a test message to confirm the component is mounting
    addTestMessage('Component mounted, establishing WebSocket connection...', 'info');
    
    // Connect to the WebSocket
    const socketUrl = window.location.protocol === 'https:' 
      ? 'wss://' + window.location.host + '/ws/pipeline/' 
      : 'ws://' + (process.env.NODE_ENV === 'development' ? 'localhost:8000' : window.location.host) + '/ws/pipeline/';
    
    console.log('Connecting to WebSocket:', socketUrl);
    
    const connectWebSocket = () => {
      if (ws.current?.readyState === WebSocket.OPEN) {
        console.log('WebSocket already connected');
        return;
      }
      
      // Clear any existing reconnect timer
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      
      console.log('Attempting WebSocket connection to:', socketUrl);
      addTestMessage(`Connecting to: ${socketUrl}`, 'info');
      
      try {
        ws.current = new WebSocket(socketUrl);
        
        // Connection opened
        ws.current.onopen = () => {
          console.log('Connected to pipeline WebSocket');
          setConnected(true);
          addTestMessage('WebSocket connection established', 'info');
          
          // Reset state when reconnecting to ensure fresh data display
          setMessages(prev => {
            // Keep only info and connection messages
            const filteredMessages = prev.filter(
              msg => msg.status === 'info' || 
                    (msg.status === 'test' && msg.message.includes('connection'))
            );
            return [...filteredMessages, { status: 'info', message: 'WebSocket connection re-established. Waiting for new data...' }];
          });
          
          // Send a message to start pipeline monitoring
          const startMessage = JSON.stringify({
            action: 'start_pipeline'
          });
          console.log('Sending start message:', startMessage);
          try {
            ws.current.send(startMessage);
            addTestMessage('Sent: start_pipeline request', 'info');
          } catch (error) {
            console.error('Error sending start message:', error);
            addTestMessage(`Error sending message: ${error.message}`, 'error');
          }
        };
    
        // Listen for messages
        ws.current.onmessage = (event) => {
          console.log('Received WebSocket message:', event.data);
          addTestMessage(`Received raw message: ${event.data.substring(0, 50)}${event.data.length > 50 ? '...' : ''}`, 'info');
          
          let data;
          try {
            data = JSON.parse(event.data);
            handleMessage(data);
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
            addTestMessage(`Error parsing message: ${error.message}`, 'error');
          }
        };
    
        // Connection closed
        ws.current.onclose = (event) => {
          console.log('Disconnected from pipeline WebSocket:', event.code, event.reason);
          setConnected(false);
          addTestMessage(`WebSocket disconnected (${event.code}): ${event.reason || 'No reason specified'}`, 'error');
          
          // Try to reconnect after a delay unless this was a clean close
          if (event.code !== 1000) {
            console.log('Scheduling reconnect attempt in 3 seconds...');
            addTestMessage('Will attempt to reconnect in 3 seconds...', 'info');
            
            reconnectTimerRef.current = setTimeout(() => {
              console.log('Attempting to reconnect...');
              connectWebSocket();
            }, 3000);
          }
        };
    
        // Connection error
        ws.current.onerror = (error) => {
          console.error('WebSocket error:', error);
          setConnected(false);
          addTestMessage(`WebSocket error: ${error.message || 'Unknown error'}`, 'error');
        };
      } catch (error) {
        console.error('Error creating WebSocket connection:', error);
        addTestMessage(`Error creating WebSocket: ${error.message}`, 'error');
        
        // Schedule a reconnect attempt
        reconnectTimerRef.current = setTimeout(() => {
          console.log('Attempting to reconnect after error...');
          connectWebSocket();
        }, 3000);
      }
    };
    
    // Initialize connection
    connectWebSocket();
    
    // Clean up the WebSocket connection when the component unmounts
    return () => {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      
      if (ws.current) {
        console.log('Closing WebSocket connection due to component unmount');
        ws.current.close();
        ws.current = null;
      }
    };
  }, []);

  // Add test button handlers (kept for functionality but not used in UI)
  const handleTestMessage = () => {
    addTestMessage('This is a manual test message from the client', 'test');
  };
  
  const handleTestHeartAttack = () => {
    addTestMessage('HEART ATTACK DETECTED - Test alert', 'output');
    setHeartAttackDetected(true);
    setHeartRate(120);
    setBloodPressure({ systolic: 150, diastolic: 95 });
  };

  // Add refresh button handler (kept for functionality but not used in UI)
  const handleRefresh = () => {
    addTestMessage('Manual refresh requested', 'info');
    
    // Clear existing state
    setDiagnosisResult(null);
    setHeartAttackDetected(false);
    setHeartRate(null);
    setBloodPressure({ systolic: null, diastolic: null });
    
    // Close and reopen the connection
    if (ws.current) {
      // Only close if it's open
      if (ws.current.readyState === WebSocket.OPEN) {
        ws.current.close();
      }
      
      // Set a timeout to reconnect after a short delay
      setTimeout(() => {
        if (ws.current?.readyState !== WebSocket.OPEN) {
          // Recreate the connection if it's not already reconnected
          const socketUrl = window.location.protocol === 'https:' 
            ? 'wss://' + window.location.host + '/ws/pipeline/' 
            : 'ws://' + (process.env.NODE_ENV === 'development' ? 'localhost:8000' : window.location.host) + '/ws/pipeline/';
          
          console.log('Reconnecting to WebSocket after refresh:', socketUrl);
          ws.current = new WebSocket(socketUrl);
          
          // Set up handlers again
          ws.current.onopen = () => {
            console.log('Reconnected to pipeline WebSocket');
            setConnected(true);
            addTestMessage('WebSocket reconnected', 'info');
            
            // Send a message to start pipeline monitoring
            try {
              ws.current.send(JSON.stringify({ action: 'start_pipeline' }));
              addTestMessage('Sent: start_pipeline request', 'info');
            } catch (error) {
              console.error('Error sending start message:', error);
              addTestMessage(`Error sending message: ${error.message}`, 'error');
            }
          };
          
          // Re-attach the message, close and error handlers
          ws.current.onmessage = (event) => {
            try {
              const data = JSON.parse(event.data);
              handleMessage(data);
            } catch (error) {
              console.error('Error parsing WebSocket message:', error);
            }
          };
          
          ws.current.onclose = () => {
            console.log('WebSocket disconnected after refresh');
            setConnected(false);
          };
          
          ws.current.onerror = (error) => {
            console.error('WebSocket error after refresh:', error);
            setConnected(false);
          };
        }
      }, 500);
    }
  };

  // Helper function to get the correct message style based on status
  const getMessageStyle = (msgStatus) => {
    switch(msgStatus) {
      case 'error': return { ...styles.message, ...styles.errorMessage };
      case 'output': return { ...styles.message, ...styles.outputMessage };
      case 'complete': return { ...styles.message, ...styles.completeMessage };
      case 'highlight': return { ...styles.message, ...styles.highlightMessage };
      case 'diagnosis': 
        return { ...styles.message, ...styles.diagnosisMessage };
      case 'info': return { ...styles.message, color: '#17a2b8' };
      case 'test': return { ...styles.message, color: '#9775fa' };
      default: return styles.message;
    }
  };

  return (
    <div style={styles.pipelineOutput}>
      <div style={styles.connectionStatus}>
        <span style={{ ...styles.statusIndicator, ...(connected ? styles.connected : styles.disconnected) }}></span>
        {connected ? 'Connected to Pipeline' : 'Disconnected from Pipeline'}
      </div>
      
      <div style={styles.mainContainer}>
        {/* Processing Status Card - Always visible */}
        <div style={styles.cardContainer}>
          <h3 style={styles.cardTitle}>Processing Status</h3>
          
          <div style={styles.spinnerContainer}>
            <div style={{
              fontSize: '60px',
              animation: 'spin 3s infinite linear',
              color: '#4285F4',
            }}>
              ⚙️
            </div>
          </div>
          
          <div style={styles.processingIndicator}>
            <div style={styles.processingLabel}>
              PCG data is being continuously monitored and analyzed
            </div>
            
            <div style={{
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              width: '100%',
              marginTop: '15px',
              opacity: 0.8,
            }}>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                width: '70px'
              }}>
                {[0, 1, 2].map(i => (
                  <div key={i} style={{
                    width: '10px',
                    height: '10px',
                    backgroundColor: '#4285F4',
                    borderRadius: '50%',
                    animation: 'pulseScale 1.4s ease-in-out infinite',
                    animationDelay: `${i * 0.2}s`
                  }}></div>
                ))}
              </div>
            </div>
          </div>
        </div>
        
        {/* Results Card - Always visible, updates with new results */}
        <div style={styles.cardContainer}>
          <h3 style={styles.cardTitle}>Heart Attack Risk Assessment</h3>
          
          {diagnosisResult ? (
            <div style={{
              ...styles.diagnosisResult,
              ...(diagnosisResult.riskLevel === 'low' ? styles.diagnosisResultLow :
                 diagnosisResult.riskLevel === 'medium' ? styles.diagnosisResultMedium :
                 diagnosisResult.riskLevel === 'high' ? styles.diagnosisResultHigh :
                 styles.diagnosisResultUnknown),
              animation: 'fadeInRight 0.5s ease-out'
            }}>
              <div style={{...styles.diagnosisIcon, animation: 'breathe 3s infinite ease-in-out'}}>
                {diagnosisResult.riskLevel === 'low' && '✅'}
                {diagnosisResult.riskLevel === 'medium' && '⚠️'}
                {diagnosisResult.riskLevel === 'high' && '🚨'}
                {diagnosisResult.riskLevel === 'unknown' && '❓'}
              </div>
              
              <p style={styles.diagnosisMessage}>
                {diagnosisResult.riskLevel === 'low' && 'LOW RISK'}
                {diagnosisResult.riskLevel === 'medium' && 'MEDIUM RISK'}
                {diagnosisResult.riskLevel === 'high' && 'HIGH RISK - URGENT ATTENTION REQUIRED'}
                {diagnosisResult.riskLevel === 'unknown' && 'UNDETERMINED RISK'}
              </p>
              
              <p style={styles.diagnosisDescription}>
                {diagnosisResult.riskLevel === 'low' && 
                  'The analysis indicates a low probability of heart attack based on the processed data. Continue regular monitoring.'}
                {diagnosisResult.riskLevel === 'medium' && 
                  'The analysis shows moderate risk indicators. Further clinical assessment is recommended.'}
                {diagnosisResult.riskLevel === 'high' && 
                  'The analysis detected high risk indicators associated with myocardial infarction. Immediate medical attention is strongly advised.'}
                {diagnosisResult.riskLevel === 'unknown' && 
                  'Unable to determine risk level from the provided data. Please consult a healthcare professional.'}
              </p>
              
              <div style={styles.monitoringAlert}>
                <div style={styles.pulsingDot}></div>
                <span>Signals are still being monitored. This assessment may change based on ongoing analysis.</span>
              </div>
            </div>
          ) : (
            <div style={styles.noResultsMessage}>
              No results available yet
            </div>
          )}
        </div>
        
        {/* Details/Console Section - Toggleable */}
        <div style={styles.cardContainer}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '15px',
          }}>
            <h3 style={{...styles.cardTitle, margin: 0, border: 'none', padding: 0}}>Technical Details</h3>
            <button 
              style={{
                ...styles.toggleButton,
                transform: showConsole ? 'scale(1.05)' : 'scale(1)',
                background: showConsole ? 'linear-gradient(135deg, #e5e5e5, #d5d5d5)' : 'linear-gradient(135deg, #f5f5f5, #e5e5e5)'
              }}
              onClick={() => setShowConsole(!showConsole)}
            >
              {showConsole ? 'Hide Console' : 'Show Console'}
            </button>
          </div>
          
          {showConsole && (
            <div style={{
              ...styles.outputConsole,
              animation: 'fadeIn 0.3s ease-out',
            }}>
              <div style={{...styles.messageContainer, maxHeight: '300px'}}>
                {messages.length === 0 ? (
                  <div style={styles.message}>No messages yet. Waiting for pipeline output...</div>
                ) : (
                  messages.map((msg, index) => (
                    <div 
                      key={index} 
                      style={getMessageStyle(msg.status)}
                    >
                      {msg.message}
                    </div>
                  ))
                )}
                <div ref={messagesEndRef} />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default PipelineOutput; 