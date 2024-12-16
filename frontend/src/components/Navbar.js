import React from "react";
import { Link } from "react-router-dom";
import "./Navbar.css";

const Navbar = () => {
  return (
    <nav className="navbar">
        <div className="logo">
          <Link to="/" className="logo-link">G-173</Link>
        </div>
        <ul className="nav-links">
          <li>
            <Link to="/" className="nav-item">Home</Link>
          </li>
          <li>
            <Link to="/features" className="nav-item">Features</Link>
          </li>
          <li>
            <Link to="/about" className="nav-item">About us</Link>
          </li>
          <li>
            <Link to="/contact" className="nav-item">Contact</Link>
          </li>
        </ul>
      </nav>
  );
};

export default Navbar;
