// import React, { useState } from "react";
// import { useNavigate } from "react-router-dom";
// import "./Admin.css";

// const Admin = () => {
//   const [selectedPhase, setSelectedPhase] = useState(null);
//   const [showLogin, setShowLogin] = useState(false);
//   const navigate = useNavigate();

//   const handleSelect = (phase) => {
//     setSelectedPhase(phase); // Set the selected phase
//     setTimeout(() => setShowLogin(true), 500); // Show login fields after transition
//   };

//   const handleLogin = () => {
//     // Redirect based on the selected phase
//     if (selectedPhase === "Phase 1") navigate("/AdminPhase1");
//     if (selectedPhase === "Phase 2") navigate("/AdminPhase2");
//     if (selectedPhase === "Phase 3") navigate("/AdminPhase3");
//   };

//   return (
//     <div className="admin-container">
//       <h2 className="admin-heading">Admin Panel</h2>

//       <div className="phase-boxes">
//         {["Phase 1", "Phase 2", "Phase 3"].map((phase, index) => (
//           <div
//             key={index}
//             className={`phase-box ${
//               selectedPhase === phase ? "selected" : selectedPhase ? "hidden" : ""
//             }`}
//           >
//             <h3>{phase}</h3>
//             <img src={`/assets/adminim.jpg`} alt="Phase" />
//             <button onClick={() => handleSelect(phase)}>Select</button>
//           </div>
//         ))}
//       </div>

//       {showLogin && (
//         <div className="login-container">
//           <h3>{selectedPhase}</h3>
//           <div className="login-box">
//             <div className="form-field">
//               <label>Username:</label>
//               <input type="text" placeholder="Enter username" />
//             </div>
//             <div className="form-field">
//               <label>Password:</label>
//               <input type="password" placeholder="Enter password" />
//             </div>
//           </div>
//           <button className="login-button" onClick={handleLogin}>
//             Login
//           </button>
//           <p>
//             Don’t have an account?{" "}
//             <span
//               className="register-link"
//               onClick={() => navigate("/register")}
//             >
//               Register
//             </span>
//           </p>
//         </div>
//       )}
//     </div>
//   );
// };

// export default Admin;

import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import "./Admin.css";


const Admin = () => {
  const [selectedPhase, setSelectedPhase] = useState(null);
  const [showLogin, setShowLogin] = useState(false);
  const [formData, setFormData] = useState({ username: "", password: "" });
  const navigate = useNavigate();

  const handleSelect = (phase) => {
    setSelectedPhase(phase);
    setTimeout(() => setShowLogin(true), 500);
  };

  const handleInputChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleLogin = async () => {
    try {
      const response = await axios.post("http://localhost:5000/login", formData);
      alert(response.data.message);
    
      if (response.data.token) {
        if (selectedPhase === "Single Face Recognition") navigate("/AdminPhase1");
        if (selectedPhase === "Group Face Recognition") navigate("/AdminPhase2");
        if (selectedPhase === "Crowd Counting") navigate("/AdminPhase3");
      }
    } catch (error) {
      alert(error.response?.data?.message || "Error logging in");
    }
    
  };

  return (
    <div className="admin-container">
      <h2 className="admin-heading">Admin Panel</h2>

      <div className="phase-boxes">
        {[
          { phase: "Single Face Recognition", image: "phase1.jpg" },
          { phase: "Group Face Recognition", image: "phase2.jpg" },
          { phase: "Crowd Counting", image: "phase3.jpg" },
        ].map((phase, index) => (
          <div
            key={index}
            className={`phase-box ${
              selectedPhase === phase.phase ? "selected" : selectedPhase ? "hidden" : ""
            }`}
          >
            
            <h3 className="phase-title">{phase.phase}</h3>

            <img
              src={require(`../assets/${phase.image}`)}
              alt={phase.phase}
            />
            
            <button onClick={() => handleSelect(phase.phase)}>Select</button>
          </div>
          
        ))}
      </div>

      {showLogin && (
        <div className="login-container">
          <h3>{selectedPhase}</h3>
          <div className="login-box">
            <div className="form-field">
              <label>Username:</label>
              <input
                type="text"
                name="username"
                placeholder="Enter username"
                value={formData.username}
                onChange={handleInputChange}
              />
            </div>
            <div className="form-field">
              <label>Password:</label>
              <input
                type="password"
                name="password"
                placeholder="Enter password"
                value={formData.password}
                onChange={handleInputChange}
              />
            </div>
          </div>
          <button className="login-button" onClick={handleLogin}>
            Login
          </button>
          <p>
            Don’t have an account?{" "}
            <span
              className="register-link"
              onClick={() => navigate("/AdminRegister")}
            >
              Register
            </span>
          </p>
        </div>
      )}
      <div className="user-register-container">
        <button
          className="user-register-button"
          onClick={() => navigate("/UserRegister")}
        >
          User Register
        </button>
      </div>
    </div>
  );
};

export default Admin;

