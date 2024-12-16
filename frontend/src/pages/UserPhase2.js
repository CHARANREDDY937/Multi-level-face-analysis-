import React from "react";
import Navbar from "../components/Navbar";
import "./UserPhase2.css";

const App = () => {
  const startRecognition = () => {
    fetch("http://localhost:5010/recognize", { method: "POST" }) // Correct fetch statement
      .then((response) => response.json())
      .then((data) => {
        console.log(data);
        const videoElement = document.getElementById("video");
        videoElement.src = "http://localhost:5010/video_feed";
        videoElement.style.display = "block";
        videoElement.style.opacity = 1;
      })
      .catch((error) => console.error("Error:", error));
  };
  
  const stopRecognition = () => {
    fetch("http://localhost:5010/stop_recognition", { method: "POST" }) // Correct fetch statement
      .then((response) => response.json())
      .then((data) => {
        console.log(data);
        const videoElement = document.getElementById("video");
        videoElement.src = "";
        videoElement.style.opacity = 0;
        setTimeout(() => {
          videoElement.style.display = "none";
        }, 300);
      })
      .catch((error) => console.error("Error:", error));
  };

  return (
    <div className="user-phase2-container">
      <Navbar />
      <div className="user-phase2-content">
        <h1>User Phase 2</h1>
        <p>Welcome to User Phase 2!</p>

        <h1>Face Recognition App</h1>
        <button onClick={startRecognition}>Start Recognition</button>
        <button onClick={stopRecognition}>Stop Recognition</button>
        <div>
          <img id="video" alt="Video Feed" />
        </div>
      </div>
    </div>
  );
};

export default App;

// import React, { useState } from "react";
// import Navbar from "../components/Navbar";
// import "./UserPhase2.css";

// const UserPhase2 = () => {
//   const [studentStatus, setStudentStatus] = useState("");
//   const [trainStatus, setTrainStatus] = useState("");
//   const [attendanceStatus, setAttendanceStatus] = useState("");
//   const [isMarkingAttendance, setIsMarkingAttendance] = useState(false);

//   const handleAddStudent = (e) => {
//     e.preventDefault();
//     const studentId = e.target.student_id.value;
//     const name = e.target.name.value;
//     const department = e.target.department.value;

//     fetch("http://localhost:5020/add_student", {
//       method: "POST",
//       headers: {
//         "Content-Type": "application/json",
//       },
//       body: JSON.stringify({
//         student_id: studentId,
//         name: name,
//         department: department,
//       }),
//     })
//       .then((response) => response.json())
//       .then((data) => setStudentStatus(data.message))
//       .catch(() => setStudentStatus("Error adding student!"));
//   };

//   const handleTrainModel = () => {
//     fetch("http://localhost:5020/train_model", {
//       method: "POST",
//     })
//       .then((response) => response.json())
//       .then((data) => setTrainStatus(data.message))
//       .catch(() => setTrainStatus("Error training model!"));
//   };

//   const handleStartAttendance = () => {
//     setIsMarkingAttendance(true);

//     fetch("http://localhost:5020/mark_attendance", {
//       method: "GET",
//     })
//       .then((response) => response.json())
//       .then((data) => {
//         document.getElementById("startAttendanceBtn").disabled = true;
//         document.getElementById("stopAttendanceBtn").disabled = false;
//         setAttendanceStatus(data.message);
//       })
//       .catch(() => {
//         document.getElementById("startAttendanceBtn").disabled = false;
//         document.getElementById("stopAttendanceBtn").disabled = true;
//         setAttendanceStatus("Error marking attendance!");
//       });
//   };

//   const handleStopAttendance = () => {
//     setIsMarkingAttendance(false);

//     fetch("http://localhost:5020/stop_marking_attendance", {
//       method: "GET",
//     })
//       .then((response) => response.json())
//       .then((data) => {
//         document.getElementById("stopAttendanceBtn").disabled = true;
//         document.getElementById("startAttendanceBtn").disabled = false;
//         setAttendanceStatus(data.message || "Attendance stopped.");
//       })
//       .catch(() => {
//         setAttendanceStatus("Error stopping attendance!");
//       });
//   };

//   return (
//     <div className="user-phase2-container">
//       <Navbar />
//       <div className="user-phase2-content">
//         <h1>Face Recognition System</h1>

//         {/* Add Student Section */}
//         {/* <section>
//           <h2>Add Student</h2>
//           <form onSubmit={handleAddStudent}>
//             <label htmlFor="student_id">Student ID:</label>
//             <input type="number" id="student_id" name="student_id" required />

//             <label htmlFor="name">Name:</label>
//             <input type="text" id="name" name="name" required />

//             <label htmlFor="department">Department:</label>
//             <input type="text" id="department" name="department" required />

//             <button type="submit">Add Student</button>
//           </form>
//           <p>{studentStatus}</p>
//         </section> */}

//         {/* Video Feed Section */}
//         <section>
//           <h2>Video Feed</h2>
//           <img
//             id="videoElement"
//             src="http://localhost:5020/video_feed"
//             alt="Video Feed"
//             style={{
//               width: "100%",
//               maxWidth: "600px",
//               border: "2px solid #ccc",
//               borderRadius: "8px",
//               marginBottom: "10px",
//             }}
//           />
//         </section>

//         {/* Training Section */}
//         {/* <section>
//           <h2>Train Model</h2>
//           <button onClick={handleTrainModel}>Train Model</button>
//           <p>{trainStatus}</p>
//         </section> */}

//         {/* Attendance Section */}
//         <section>
//           <h2>Mark Attendance</h2>
//           <button
//             id="startAttendanceBtn"
//             onClick={handleStartAttendance}
//             disabled={isMarkingAttendance}
//           >
//             Start Marking Attendance
//           </button>
//           <button
//             id="stopAttendanceBtn"
//             onClick={handleStopAttendance}
//             disabled={!isMarkingAttendance}
//           >
//             Stop Marking Attendance
//           </button>
//           <p>{attendanceStatus}</p>
//         </section>
//       </div>
//     </div>
//   );
// };

// export default UserPhase2;

