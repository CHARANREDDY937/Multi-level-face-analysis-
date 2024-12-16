import React from 'react';
import '../pages/About.css';

function About() {
  return (
    <div className="about-container">
      <h1>About Our Team</h1>
      <p className="team-description">
        We are a group of passionate innovators dedicated to creating cutting-edge AI solutions.
      </p>
      
      <div className="team-grid">
        
        <div className="team-card">
          <h2 className="team-name">Pranav Kumar Jangam</h2>
          <p className="team-role">23BD1A6722</p>
          <p className="team-bio">
            CSD-E1
          </p>
        </div>

        <div className="team-card">
          <h2 className="team-name">Voore Risheel Kumar</h2>
          <p className="team-role">23BD1A05BV</p>
          <p className="team-bio">
           CSE-E4
          </p>
        </div>
        <div className="team-card">
          <h2 className="team-name">Mohammad Nida Madeeha</h2>
          <p className="team-role">23BD1A05B4
          </p>
          <p className="team-bio">
           CSE-E4
          </p>
        </div>
        <div className="team-card">
          <h2 className="team-name">Bannuru Charan Reddy</h2>
          <p className="team-role">23BD1A05A5
          </p>
          <p className="team-bio">
           CSE-E4
          </p>
        </div>
        <div className="team-card">
          <h2 className="team-name">Bheemireddy Charan Sai Reddy</h2>
          <p className="team-role">23BD1A05AE
          </p>
          <p className="team-bio">
           CSE-E4
          </p>
        </div>
        
        <div className="team-card">
          <h2 className="team-name">Kolipyaka Vyshnavi</h2>
          <p className="team-role">23BD1A05AW</p>
          <p className="team-bio">
           CSE-E4
          </p>
        </div>
        <div className="team-card">
          <h2 className="team-name">Tankasala Akshaya</h2>
          <p className="team-role">23BD1A05BK</p>
          <p className="team-bio">
           CSE-E4
          </p>
        </div>
      </div>
    </div>
  );
}

export default About;
