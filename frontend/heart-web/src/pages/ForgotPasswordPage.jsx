// src/pages/ForgotPasswordPage.jsx
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './LoginPage.css'; // Reuse the same CSS or create a new one

function ForgotPasswordPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const handleForgotPassword = () => {
    // Clear previous error/message
    setError('');
    setMessage('');

    // Simple validation: email should not be empty
    if (!email.trim()) {
      setError('Please enter your email.');
      return;
    }

    // Your API call or logic to send the user their password or reset link
    console.log('Requesting password for:', email);
    setMessage('If this email exists, a password has been sent to it.');
  };

  return (
    <div className="login-container">
      <div className="login-box">
        <h1>Heart Attack Detection</h1>
        <h2>Forgot Password</h2>

        {error && <p className="error-message">{error}</p>}
        {message && <p className="success-message">{message}</p>}

        <label htmlFor="email">Email</label>
        <input
          id="email"
          type="email"
          placeholder="Enter your email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />

        <button onClick={handleForgotPassword}>Submit</button>

        {/* Back to Login Button */}
        <button className="back-button" onClick={() => navigate('/')}>
          Back to Sign In
        </button>
      </div>
    </div>
  );
}

export default ForgotPasswordPage;
