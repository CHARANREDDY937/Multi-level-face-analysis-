// import React, { useState, useEffect } from "react";
// import Navbar from "../components/Navbar";
// import "./AdminPhase2.css";

// const AdminPhase2 = () => {
//     const [notification, setNotification] = useState(null); // Updated state to handle object consistently
//     const [logMessages, setLogMessages] = useState([]);
//     const [formData, setFormData] = useState({
//         addStudentId: "",
//         addStudentName: "",
//         deleteStudentId: "",
//         captureStudentId: "",
//         captureStudentName: "",
//     });

//     // Cleanup flag for async operations
//     useEffect(() => {
//         let isMounted = true;
//         return () => {
//             isMounted = false;
//         };
//     }, []);

//     // Handle input change for form fields
//     const handleInputChange = (e) => {
//         const { name, value } = e.target;
//         setFormData((prevState) => ({
//             ...prevState,
//             [name]: value,
//         }));
//     };

//     // Show notification
//     const showNotification = (message, isError = false) => {
//         setNotification({ message, isError });
//         setTimeout(() => {
//             setNotification(null); // Clear the notification after 3 seconds
//         }, 3000);
//     };

//     // Log message
//     const logMessage = (message) => {
//         setLogMessages((prevMessages) => [...prevMessages, `${new Date().toLocaleTimeString()} - ${message}`]);
//     };

//     // Send request to server
//     const sendRequest = async (url, data, logMsg) => {
//         logMessage(logMsg);
//         try {
//             const response = await fetch(url, {
//                 method: "POST",
//                 headers: { "Content-Type": "application/json" },
//                 body: JSON.stringify(data),
//             });

//             if (!response.ok) {
//                 const errorText = await response.text(); // Get the error page or message
//                 throw new Error(errorText);
//             }

//             const result = await response.json();
//             showNotification(result.message);
//             logMessage(result.message);
//         } catch (error) {
//             showNotification(`An error occurred: ${error.message}`, true);
//             logMessage(`Error: ${error.message}`);
//         }
//     };

//     // Handlers for form submissions
//     const handleAddStudent = () => {
//         sendRequest(
//             "/add_student",
//             { student_id: formData.addStudentId, student_name: formData.addStudentName },
//             "Adding student..."
//         );
//     };

//     const handleDeleteStudent = () => {
//         sendRequest(
//             "/delete_student",
//             { student_id: formData.deleteStudentId },
//             "Deleting student..."
//         );
//     };

//     const handleCaptureImages = () => {
//         sendRequest(
//             "/capture_images",
//             { student_id: formData.captureStudentId, student_name: formData.captureStudentName },
//             "Capturing images..."
//         );
//     };

//     const handleTrainModel = () => {
//         sendRequest("/train_model", {}, "Training model...");
//     };

//     const handleMarkAttendance = () => {
//         sendRequest("/mark_attendance", {}, "Starting attendance...");
//     };

//     return (
//         <div>
//             <Navbar />
//             <div className="container">
//                 <h1>Attendance System Dashboard</h1>

//                 {/* Add Student */}
//                 <form id="add-student-form">
//                     <h2>Add Student</h2>
//                     <label htmlFor="add-student-id">Student ID:</label>
//                     <input
//                         type="text"
//                         id="add-student-id"
//                         name="addStudentId"
//                         value={formData.addStudentId}
//                         onChange={handleInputChange}
//                         placeholder="Enter Student ID"
//                         required
//                     />
//                     <label htmlFor="add-student-name">Student Name:</label>
//                     <input
//                         type="text"
//                         id="add-student-name"
//                         name="addStudentName"
//                         value={formData.addStudentName}
//                         onChange={handleInputChange}
//                         placeholder="Enter Student Name"
//                         required
//                     />
//                     <button type="button" onClick={handleAddStudent}>Add Student</button>
//                 </form>

//                 {/* Delete Student */}
//                 <form id="delete-student-form">
//                     <h2>Delete Student</h2>
//                     <label htmlFor="delete-student-id">Student ID:</label>
//                     <input
//                         type="text"
//                         id="delete-student-id"
//                         name="deleteStudentId"
//                         value={formData.deleteStudentId}
//                         onChange={handleInputChange}
//                         placeholder="Enter Student ID to Delete"
//                         required
//                     />
//                     <button type="button" onClick={handleDeleteStudent}>Delete Student</button>
//                 </form>

//                 {/* Capture Images */}
//                 <form id="capture-images-form">
//                     <h2>Capture Images</h2>
//                     <label htmlFor="capture-student-id">Student ID:</label>
//                     <input
//                         type="text"
//                         id="capture-student-id"
//                         name="captureStudentId"
//                         value={formData.captureStudentId}
//                         onChange={handleInputChange}
//                         placeholder="Enter Student ID"
//                         required
//                     />
//                     <label htmlFor="capture-student-name">Student Name:</label>
//                     <input
//                         type="text"
//                         id="capture-student-name"
//                         name="captureStudentName"
//                         value={formData.captureStudentName}
//                         onChange={handleInputChange}
//                         placeholder="Enter Student Name"
//                         required
//                     />
//                     <button type="button" onClick={handleCaptureImages}>Capture Images</button>
//                 </form>

//                 {/* Train Model */}
//                 <form id="train-model-form">
//                     <h2>Train Model</h2>
//                     <button type="button" onClick={handleTrainModel}>Train Model</button>
//                 </form>

//                 {/* Mark Attendance */}
//                 <form id="mark-attendance-form">
//                     <h2>Mark Attendance</h2>
//                     <button type="button" onClick={handleMarkAttendance}>Start Attendance</button>
//                 </form>

//                 {/* Logs Section */}
//                 <div id="log-container">
//                     {logMessages.map((log, index) => (
//                         <div key={index}>{log}</div>
//                     ))}
//                 </div>
//             </div>

//             {/* Notification */}
//             {notification && (
//                 <div
//                     className={`notification ${notification.isError ? "error" : ""} show`}
//                     style={{ position: "fixed", bottom: "20px", right: "20px" }}
//                 >
//                     {notification.message || ""}
//                 </div>
//             )}
//         </div>
//     );
// };

// export default AdminPhase2;


// import React, { useState } from 'react';
// import './AdminPhase2.css';

// function AdminPhase2() {
//   const [log, setLog] = useState('');

//   const logMessage = (message) => {
//     setLog((prevLog) => `${prevLog}${message}\n`);  // Fixed template literal
//   };

//   const sendRequest = async (endpoint, data = {}) => {
//     try {
//       const response = await fetch(endpoint, {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify(data),
//       });

//       const result = await response.json();
//       logMessage(result.message || result.error || 'Operation completed');
//     } catch (error) {
//       logMessage(`Error: ${error.message}`);  // Fixed template literal for error message
//     }
//   };

//   const handleAddStudent = () => {
//     const studentId = prompt('Enter Student ID:');
//     const studentName = prompt('Enter Student Name:');
//     if (studentId && studentName) {
//       sendRequest('/add_student', { student_id: studentId, student_name: studentName });
//     } else {
//       logMessage('Failed to capture input');
//     }
//   };

//   const handleDeleteStudent = () => {
//     const studentId = prompt('Enter Student ID to delete:');
//     if (studentId) {
//       sendRequest('/delete_student', { student_id: studentId });
//     } else {
//       logMessage('Failed to capture input');
//     }
//   };

//   const handleCaptureImages = () => {
//     const studentId = prompt('Enter Student ID:');
//     const studentName = prompt('Enter Student Name:');
//     if (studentId && studentName) {
//       sendRequest('/capture_images', { student_id: studentId, student_name: studentName });
//     } else {
//       logMessage('Failed to capture input');
//     }
//   };

//   const handleTrainModel = () => {
//     sendRequest('/train_model');
//   };

//   const handleMarkAttendance = () => {
//     sendRequest('/mark_attendance');
//   };

//   return (
//     <div className="container">
//       <h1>Attendance System Dashboard</h1>
//       <div className="button-container">
//         <button onClick={handleAddStudent}>Add Student</button>
//         <button onClick={handleDeleteStudent}>Delete Student</button>
//         <button onClick={handleCaptureImages}>Capture Images</button>
//         <button onClick={handleTrainModel}>Train Model</button>
//         <button onClick={handleMarkAttendance}>Mark Attendance</button>
//       </div>
//       <div className="log-container">
//         <pre>{log}</pre>
//       </div>
//     </div>
//   );
// }

// export default AdminPhase2;



import React from "react";
import Navbar from "../components/Navbar";
import "./AdminPhase2.css";

const AdminPhase2 = () => {
  const startRecognition = () => {
    fetch("http://localhost:5010/recognize", { method: "POST" })
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
    fetch("http://localhost:5010/stop_recognition", { method: "POST" })
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
    <div className="admin-phase2-container">
      <Navbar />
      <div className="admin-phase2-content">
        <h1>Admin Phase 2</h1>
        <p>Welcome to Admin Phase 2!</p>

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

export default AdminPhase2;
