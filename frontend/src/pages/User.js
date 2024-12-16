// import React, { useState } from "react";
// import { useNavigate } from "react-router-dom";
// import "./User.css";

// const User = () => {
//   const [selectedPhase, setSelectedPhase] = useState(null);
//   const [showLogin, setShowLogin] = useState(false);
//   const navigate = useNavigate();

//   const handleSelect = (phase) => {
//     setSelectedPhase(phase); 
//     setTimeout(() => setShowLogin(true), 500); 
//   };

//   const handleLogin = () => {
    
//     if (selectedPhase === "Phase 1") navigate("/userphase1");
//     if (selectedPhase === "Phase 2") navigate("/userphase2");
//     if (selectedPhase === "Phase 3") navigate("/userphase3");
//   };

//   return (
//     <div className="user-container">
//       <h2 className="user-heading">User Panel</h2>

//       <div className="phase-boxes">
//         {["Phase 1", "Phase 2", "Phase 3"].map((phase, index) => (
//           <div
//             key={index}
//             className={`phase-box ${
//               selectedPhase === phase ? "selected" : selectedPhase ? "hidden" : ""
//             }`}
//           >
//             <h3>{phase}</h3>
//             <img src={`/assets/userim.jpg`} alt="Phase" />
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
//               onClick={() => navigate("/userregister")}
//             >
//               Register
//             </span>
//           </p>
//         </div>
//       )}
//     </div>
//   );
// };

// export default User;
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import "./User.css";

const User = () => {
  const [selectedPhase, setSelectedPhase] = useState(null);
  const [showLogin, setShowLogin] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleSelect = (phase) => {
    setSelectedPhase(phase);
    setTimeout(() => setShowLogin(true), 500);
  };

  const handleLogin = async () => {
    if (!username || !password) {
      setError("Please fill in all fields.");
      return;
    }

    try {
      const response = await axios.post("http://localhost:5003/login1", {
        username,
        password,
      });
      const { token } = response.data;

      // Store the JWT token in local storage
      localStorage.setItem("token", token);

      // Redirect based on the selected phase
      if (selectedPhase === "Single Face Recognition") navigate("/userphase1");
      if (selectedPhase === "Group Face Recognition") navigate("/userphase2");
      if (selectedPhase === "Crowd Counting") navigate("/userphase3");
    } catch (err) {
      setError(err.response?.data?.message || "Error logging in.");
    }
  };

  return (
    <div className="user-container">
      <h2 className="user-heading">User Panel</h2>

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
          {error && <p className="error-message">{error}</p>}
          <div className="login-box">
            <div className="form-field">
              <label>Username:</label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter username"
              />
            </div>
            <div className="form-field">
              <label>Password:</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter password"
              />
            </div>
          </div>
          <button className="login-button" onClick={handleLogin}>
            Login
          </button>
          {/* <p>
            Don’t have an account?{" "}
            <span
              className="register-link"
              onClick={() => navigate("/userregister")}
            >
              Register
            </span>
          </p> */}
        </div>
      )}
    </div>
  );
};

export default User;
