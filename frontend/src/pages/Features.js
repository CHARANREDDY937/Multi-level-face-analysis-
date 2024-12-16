import React from 'react';
import '../pages/Features.css';

function Features() {
  return (
    
    <div className="features-container">
      <h1>Features of Our Models</h1>
      <p className="features-description">
        Our models are designed to provide cutting-edge AI solutions for various identification and analysis needs.
      </p>

      <div className="features-grid">
        <div className="feature-card">
          <img
            src="https://img.freepik.com/premium-photo/closeup-businesswoman39s-face-showing-determination-focus-while-working-project-with-clean-professional_1229213-60777.jpg"
            alt="Single Face Recognition"
            className="feature-image"
          />
          <h2 className="feature-title">Single Face Recognition</h2>
          <p className="feature-description">
          Recognition model is a Flask-based web application provides a comprehensive pipeline for capturing, training, and recognizing faces for single-person identification. It integrates with MongoDB to store face images and CNN model data, facilitating a complete face recognition workflow.
          </p>
        </div>

        <div className="feature-card">
          <img
            src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQzksC2Zv417zE-ejxSZD8Zj9CZhKGiVAsPUQ&s"
            alt="Group Face Recognition"
            className="feature-image"
          />
          <h2 className="feature-title">Group Face Recognition</h2>
          <p className="feature-description">
          Group Face Recognition model is a Flask-based web application performs real-time face detection and recognition for multiple individuals using a pre-trained Convolutional Neural Network (CNN). The system identifies faces and matches them to a predefined list of users, displaying the recognized names on the live video stream.
          </p>
        </div>

        
        <div className="feature-card">
          <img
            src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS_6mhehB2KDAIq33ajE7n5J4NSXBUJZTOnYQ&s"
            alt="Crowd Analysis"
            className="feature-image"
          />
          <h2 className="feature-title">Crowd Analysis</h2>
          <p className="feature-description">
          Crowd counting model is a Flask-based web application provides real-time crowd counting using a live webcam feed and the LWCC (Lightweight Crowd Counting) model. The application captures video frames, processes them with the LWCC model to estimate the number of people, and displays the crowd count overlayed on the video stream.
          </p>
        </div>
      </div>
    </div>
  );
}

export default Features;
