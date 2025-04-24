// src/pages/FancyDashboardPage.jsx
import React from "react";
import "./FancyDashboardPage.css";
import PipelineOutput from "../components/PipelineOutput";

function FancyDashboardPage() {
  // Example user data
  const userName = "John Doe";
  const userAvatar = "https://via.placeholder.com/50.png?text=JD";

  // Example vitals
  const heartRate = 120; // in bpm
  const bloodPressureSystolic = 120; // mmHg
  const bloodPressureDiastolic = 80; // mmHg

  // Example: if a heart attack is detected
  const isHeartAttack = true;
  const timeToAttack = 5; // minutes

  // Example instructions
  const instructions = [
    "Call emergency services immediately and explain the situation clearly.",
    "Sit down and remain calm to reduce strain on your heart.",
    "Chew and swallow a 325 mg aspirin if available and you are not allergic.",
    "Take nitroglycerin if prescribed by placing one dose under your tongue.",
    "Loosen tight clothing to help you breathe more easily.",
  ];

  return (
    <div className="fancy-dashboard">
      {/* Left sidebar (unchanged) */}
      <aside className="fancy-sidebar">
        <div className="sidebar-user-info">
          <img src={userAvatar} alt="User Avatar" className="sidebar-avatar" />
          <div className="sidebar-username">{userName}</div>
        </div>
        <nav className="sidebar-nav">
          <ul>
            <li className="active">Overview</li>
            <li>History</li>
          </ul>
        </nav>
      </aside>

      {/* Main content */}
      <main className="fancy-main">
        <header className="fancy-header">
          <div className="header-left">
            <h1>Heart Health Overview</h1>
            <p className="subtitle">
              Welcome back, <strong>{userName}</strong>
            </p>
          </div>
        </header>

        {/* 2:1 vertical split */}
        <div className="layout-container">
          {/* TOP (2fr) */}


          {/* BOTTOM (1fr): Pipeline Output Section */}
          <div className="bottom-section">
            <PipelineOutput />
          </div>
        </div>
      </main>
    </div>
  );
}

export default FancyDashboardPage;
