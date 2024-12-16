// import React from "react";
// import Navbar from "../components/Navbar";
// import "./UserPhase1.css";

// const UserPhase1 = () => {
//   return (
//     <div className="user-phase1-container">
//       <Navbar />
//       <div className="user-phase1-content">
//         <h1>User Phase 1</h1>
//         <p>Welcome to User Phase 1!</p>
//       </div>
//     </div>
//   );
// };

// export default UserPhase1;
import React from "react";
import Navbar from "../components/Navbar";
import "./UserPhase1.css";




const UserPhase1 = () => {
  const handleAuthenticate = async () => {
    try {
      const response = await fetch("http://127.0.0.1:5000/recognize_faces", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (response.ok) {
        const data = await response.json();
        alert(data.message || "Face recognition completed successfully.");
      } else {
        const errorData = await response.json();
        alert(errorData.message || "Error occurred during face recognition.");
      }
    } catch (error) {
      alert("An error occurred while connecting to the server.");
      console.error(error);
    }
  };
  return (
    <div className="user-phase1-container">
      <Navbar />
      <div className="user-phase1-content">
        <h1>Single Face Recognition</h1>
        <p>Welcome to User Phase 1!</p>
        
        <button className="admin-box" onClick={handleAuthenticate}>
          <img src={require('../assets/admin3.jpg')} alt="Authenticate" />
          <h3>Authenticate</h3>
        </button>
      </div>
    </div>
  );
};

export default UserPhase1;
