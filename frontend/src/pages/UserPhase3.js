import React, { useState } from "react";
import "./UserPhase3.css";

const UserPhase3 = () => {
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

export default UserPhase3;
