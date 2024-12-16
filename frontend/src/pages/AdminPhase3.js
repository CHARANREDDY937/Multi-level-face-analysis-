// import React, { useState } from 'react';

// import './AdminPhase3.css';

// const AdminPhase3 = () => {
//   const [selectedFile, setSelectedFile] = useState(null);
//   const [action, setAction] = useState('');

//   const handleFileChange = (e) => {
//     setSelectedFile(e.target.files[0]);
//   };

//   const handleAction = (actionType) => {
//     setAction(actionType);
//     setSelectedFile(null); // Reset the file input for a new action
//   };

//   return (
//     <div className="admin-phase3">
      
//       <div className="admin-welcome">
//         <h1>Welcome back Admin!!</h1>
//       </div>

//       <div className="admin-buttons">
//         <button className="admin-button" onClick={() => handleAction('cam')}>
//           <img src={require('../assets/cam.jpg')} alt="Count by Cam" />
//           <h3>Count by Cam</h3>
//         </button>

//         <button className="admin-button" onClick={() => handleAction('image')}>
//           <img src={require('../assets/image.jpg')} alt="Count Crowd in an pic" />
//           <h3>Count Crowd in an Image</h3>
//         </button>

//         <button className="admin-button" onClick={() => handleAction('video')}>
//           <img src={require('../assets/video.jpg')} alt="Count Crowd in a Video" />
//           <h3>Count Crowd in a Video</h3>
//         </button>
//       </div>

//       {action === 'image' && (
//         <div className="file-upload">
//           <h2>Upload an Image</h2>
//           <input type="file" accept="image/*" onChange={handleFileChange} />
//           {selectedFile && <p>Selected File: {selectedFile.name}</p>}
//         </div>
//       )}

//       {action === 'video' && (
//         <div className="file-upload">
//           <h2>Upload a Video</h2>
//           <input type="file" accept="video/*" onChange={handleFileChange} />
//           {selectedFile && <p>Selected File: {selectedFile.name}</p>}
//         </div>
//       )}
//     </div>
//   );
// };

// export default AdminPhase3;

import React, { useState } from "react";
import "./UserPhase3.css";

const AdminPhase3= () => {
  const [videoFeed, setVideoFeed] = useState("");

  const serverUrl = "http://127.0.0.1:5002"; // Replace with your Flask server IP

  const startFeed = async () => {
    try {
      const response = await fetch(`${serverUrl}/start`, { method: "POST" });
      const data = await response.json();
      console.log("Feed started:", data);
      setVideoFeed(`${serverUrl}/video_feed`);
    } catch (error) {
      console.error("Error starting feed:", error);
    }
  };

  const stopFeed = async () => {
    try {
      const response = await fetch(`${serverUrl}/stop`, { method: "POST" });
      const data = await response.json();
      console.log("Feed stopped:", data);
      setVideoFeed("");
    } catch (error) {
      console.error("Error stopping feed:", error);
    }
  };

  return (
    <div className="user-phase3-container">
      <h1>Real-Time Crowd Counting</h1>
      <div className="button-container">
        <button onClick={startFeed}>Start</button>
        <button onClick={stopFeed}>Stop</button>
      </div>
      <div className="video-container">
        {videoFeed ? (
          <img src={videoFeed} alt="Video feed will appear here" />
        ) : (
          <p>Video feed will appear here</p>
        )}
      </div>
    </div>
  );
};

export default AdminPhase3;