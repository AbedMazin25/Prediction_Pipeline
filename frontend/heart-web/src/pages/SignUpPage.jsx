// src/pages/SignUpPage.jsx
import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
// Import the same CSS used by your Login page
import "./LoginPage.css";

function SignUpPage() {
  const navigate = useNavigate();

  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [address, setAddress] = useState("");
  const [dob, setDob] = useState("");
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const handleSignUp = (e) => {
    e.preventDefault();
    // TODO: Add sign-up logic (API call, validation, etc.)
    console.log("Signing up with:", {
      firstName,
      lastName,
      address,
      dob,
      email,
      username,
      password,
    });
    // On success, navigate to login or dashboard:
    // navigate("/login");
  };

  return (
    <div className="login-container">
      <div className="login-box">
        <h1>Heart Attack Detection</h1>
        <h2>Sign Up</h2>

        <form onSubmit={handleSignUp}>
          {/* First Name */}
          <label htmlFor="firstName">First Name</label>
          <input
            id="firstName"
            type="text"
            placeholder="John"
            value={firstName}
            onChange={(e) => setFirstName(e.target.value)}
            required
          />

          {/* Last Name */}
          <label htmlFor="lastName">Last Name</label>
          <input
            id="lastName"
            type="text"
            placeholder="Doe"
            value={lastName}
            onChange={(e) => setLastName(e.target.value)}
            required
          />

          {/* Address */}
          <label htmlFor="address">Address</label>
          <input
            id="address"
            type="text"
            placeholder="123 Main St"
            value={address}
            onChange={(e) => setAddress(e.target.value)}
          />

          {/* Date of Birth */}
          <label htmlFor="dob">Date of Birth</label>
          <input
            id="dob"
            type="date"
            value={dob}
            onChange={(e) => setDob(e.target.value)}
            required
          />

          {/* Email */}
          <label htmlFor="email">Email</label>
          <input
            id="email"
            type="email"
            placeholder="john.doe@example.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />

          {/* Username */}
          <label htmlFor="username">Username</label>
          <input
            id="username"
            type="text"
            placeholder="john_doe"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
          />

          {/* Password */}
          <label htmlFor="password">Password</label>
          <input
            id="password"
            type="password"
            placeholder="********"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />

          {/* Submit Button */}
          <button type="submit">Sign Up</button>
        </form>

        <div className="links">
          <p>
            Already have an account?{" "}
            <Link to="/login">Login</Link>
          </p>
        </div>
      </div>
    </div>
  );
}

export default SignUpPage;
