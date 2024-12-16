// import React, { useState } from "react";
// import { useNavigate } from "react-router-dom";
// import "./Register.css";

// const Register = () => {
//   const [username, setUsername] = useState("");
//   const [password, setPassword] = useState("");
//   const [confirmPassword, setConfirmPassword] = useState("");
//   const [phase, setPhase] = useState("Phase 1");
//   const navigate = useNavigate();

//   const handleRegister = async (e) => {
//     e.preventDefault();

//     if (password !== confirmPassword) {
//       alert("Passwords do not match!");
//       return;
//     }

//     // Here, you can make an API call to store the registration data
//     try {
//       const response = await fetch("http://localhost:5000/register", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ username, password, phase }),
//       });

//       const data = await response.json();

//       if (response.ok) {
//         alert("Registration successful!");
//         navigate("/login"); // Redirect to login page
//       } else {
//         alert(data.message || "Registration failed!");
//       }
//     } catch (error) {
//       console.error("Error during registration:", error);
//       alert("Something went wrong. Please try again.");
//     }
//   };

//   return (
//     <div className="register-container">
//       <h2>Register</h2>
//       <div className="register-box">
//         <form onSubmit={handleRegister}>
//           <div className="form-field">
//             <label>Username:</label>
//             <input
//               type="text"
//               placeholder="Enter username"
//               value={username}
//               onChange={(e) => setUsername(e.target.value)}
//               required
//             />
//           </div>

//           <div className="form-field">
//             <label>Password:</label>
//             <input
//               type="password"
//               placeholder="Enter password"
//               value={password}
//               onChange={(e) => setPassword(e.target.value)}
//               required
//             />
//           </div>

//           <div className="form-field">
//             <label>Confirm Password:</label>
//             <input
//               type="password"
//               placeholder="Confirm password"
//               value={confirmPassword}
//               onChange={(e) => setConfirmPassword(e.target.value)}
//               required
//             />
//           </div>

//           <div className="form-field">
//             <label>Select Phase:</label>
//             <select
//               value={phase}
//               onChange={(e) => setPhase(e.target.value)}
//               required
//             >
//               <option value="Phase 1">Phase 1</option>
//               <option value="Phase 2">Phase 2</option>
//               <option value="Phase 3">Phase 3</option>
//             </select>
//           </div>

//           <button type="submit" className="register-button">
//             Register
//           </button>
//         </form>
//         <p>
//           Already have an account?{" "}
//           <span className="link" onClick={() => navigate("/login")}>
//             Login here
//           </span>
//         </p>
//       </div>
//     </div>
//   );
// };

// export default Register;
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import "./Register.css";

const Register = () => {
  const [formData, setFormData] = useState({ username: "", password: "" });
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleInputChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleRegister = async () => {
    if (!formData.username || !formData.password) {
      setError("Please fill in all fields.");
      return;
    }

    try {
      const response = await axios.post("http://localhost:5000/register", formData);
      alert(response.data.message);
      navigate("/admin"); // Redirect to Admin panel after successful registration
    } catch (err) {
      setError(err.response?.data?.message || "Error registering user.");
    }
  };

  return (
    <div className="register-container">
      
      {error && <p className="error-message">{error}</p>}
      <div className="register-box">
      <h2>Register</h2>
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
      <button className="register-button" onClick={handleRegister}>
        Register
      </button>
      <p className="para">
        Already have an account?{" "}
        <span className="login-link" onClick={() => navigate("/admin")}>
          Login here
        </span>
      </p>
    </div>
  );
};

export default Register;
