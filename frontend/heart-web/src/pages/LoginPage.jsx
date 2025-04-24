// src/pages/LoginPage.jsx
import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom'; // <-- Import useNavigate
import './LoginPage.css';

function LoginPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  // For programmatic navigation:
  const navigate = useNavigate();

  const handleLogin = () => {
    // Clear any previous error
    setError('');

    // Simple validation: check if fields are empty
    if (!username.trim() || !password.trim()) {
      setError('Please fill in both fields.');
      return;
    }

    // Your login logic here (API call, etc.)
    console.log('Logging in with:', { username, password });

    // If login is successful, navigate to the dashboard:
    navigate('/dashboard');
  };

  return (
    <div className="login-container">
      <div className="login-box">
        <h1>Heart Attack Detection</h1>
        <h2>Login</h2>

        {error && <p className="error-message">{error}</p>}

        <label htmlFor="username">Username</label>
        <input
          id="username"
          type="text"
          placeholder="Enter your username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />

        <label htmlFor="password">Password</label>
        <input
          id="password"
          type="password"
          placeholder="Enter your password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        <button onClick={handleLogin}>Login</button>

        <div className="links">
          <Link to="/forgot-password">Forgot Password?</Link>
          <p>
            Don’t have an account? <Link to="/signup">Sign Up</Link>
          </p>
        </div>
      </div>
    </div>
  );
}

export default LoginPage;
