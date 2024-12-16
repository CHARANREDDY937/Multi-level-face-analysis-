import React, { useState, useEffect } from 'react';
import { Link } from "react-router-dom";

import userImage from "../assets/userim.webp";
import "./Home.css";

const Home = () => {
  const [greeting, setGreeting] = useState('');
  useEffect(() => {
    const hours = new Date().getHours();
    if (hours < 12) setGreeting('Good Morning');
    else if (hours < 18) setGreeting('Good Afternoon');
    else setGreeting('Good Evening');
  }, []);
  return (
    <div className="home-container">
      <h1 className="main-heading">{greeting}, Welcome to the Facial Authentication System</h1>
        <p className="para">
          Unlock a new era of seamless identification and security. Our advanced AI ensures accuracy, speed, and personalized experiences like never before.
        </p>
      <div className="options-container">
        <div className="box user-box">
        <img src="https://t3.ftcdn.net/jpg/00/65/75/68/240_F_65756860_GUZwzOKNMUU3HldFoIA44qss7ZIrCG8I.jpg" alt="Admin" className="box-image" />
        <h2>Admin</h2>
        <Link to="/admin">
          <button className="btn box-button">Enter</button>
        </Link>
      </div>

      <div className="box user-box">
        <img src={userImage} alt="User" className="box-image" />
        <h2>User</h2>
        <Link to="/user">
          <button className="btn box-button">Enter</button>
        </Link>
      </div>
      </div>
    </div>
  );
};

export default Home;
