// import React, { useState } from "react";
// import "./UserRegister.css";

// const UserRegister = () => {
//   const [formData, setFormData] = useState({
//     username: "",
//     password: "",
//     confirmPassword: "",
//     phase: "Phase 1",
//   });

//   const handleChange = (e) => {
//     setFormData({ ...formData, [e.target.name]: e.target.value });
//   };

//   const handleSubmit = (e) => {
//     e.preventDefault();
//     // Add MongoDB logic here
//     console.log(formData);
//   };

//   return (
//     <div className="register-container">
//       <div className="register-box">
//       <h2>User Registration</h2>
//       <form onSubmit={handleSubmit}>
//         <div className="form-field">
//           <label>Username:</label>
//           <input
//             type="text"
//             name="username"
//             value={formData.username}
//             onChange={handleChange}
//             placeholder="Enter username"
//           />
//         </div>
//         <div className="form-field">
//           <label>Password:</label>
//           <input
//             type="password"
//             name="password"
//             value={formData.password}
//             onChange={handleChange}
//             placeholder="Enter password"
//           />
//         </div>
//         <div className="form-field">
//           <label>Confirm Password:</label>
//           <input
//             type="password"
//             name="confirmPassword"
//             value={formData.confirmPassword}
//             onChange={handleChange}
//             placeholder="Confirm password"
//           />
//         </div>
//         <div className="form-field">
//           <label>Phase:</label>
//           <select
//             name="phase"
//             value={formData.phase}
//             onChange={handleChange}
//           >
//             <option>Phase 1</option>
//             <option>Phase 2</option>
//             <option>Phase 3</option>
//           </select>
//         </div>
//         <button type="submit">Register</button>
//       </form>
//       </div>
//     </div>
//   );
// };

// export default UserRegister;

import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import "./Register.css";

const Register1 = () => {
  const [formData, setFormData] = useState({ username: "", password: "" });
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleInputChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  // const handleRegister = async () => {
  //   if (!formData.username || !formData.password) {
  //     setError("Please fill in all fields.");
  //     return;
  //   }

  //   try {
  //     const response = await axios.post("http://localhost:5000/register1", formData);
  //     alert(response.data.message);
  //     navigate("/admin"); // Redirect to Admin panel after successful registration
  //   } catch (err) {
  //     setError(err.response?.data?.message || "Error registering user.");
  //   }
  // };
  const handleRegister = async () => {
    if (!formData.username || !formData.password) {
      setError("Please fill in all fields.");
      return;
    }
  
    try {
      const response = await axios.post("http://localhost:5003/register1", formData); // Updated port to 5003
      alert(response.data.message);
      navigate("/admin"); // Redirect to Admin panel after successful registration
    } catch (err) {
      // Display meaningful error messages
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
      
    </div>
  );
};

export default Register1;
