// src/App.js
import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import LoginPage from "./pages/LoginPage";
import SignUpPage from "./pages/SignUpPage";
import ForgotPasswordPage from "./pages/ForgotPasswordPage";
import FancyDashboardPage from "./pages/FancyDashboardPage"; // Import your DashboardPage

function App() {
  return (
    <Router>
      <Routes>
        {/* Login page */}
        <Route path="/login" element={<LoginPage />} />

        {/* Sign-up page */}
        <Route path="/signup" element={<SignUpPage />} />

        {/* Forgot Password page */}
        <Route path="/forgot-password" element={<ForgotPasswordPage />} />

        {/* Dashboard page */}
        <Route path="/dashboard" element={<FancyDashboardPage />} />

        {/* Redirect "/" to login */}
        <Route path="/" element={<LoginPage />} />
      </Routes>
    </Router>
  );
}

export default App;
