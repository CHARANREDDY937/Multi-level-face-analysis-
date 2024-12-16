import React from 'react';
import './Contact.css';

function Contact() {
  return (
    <div className="contact-container">
      <h1>Contact Us</h1>
      <p className="contact-description">
        If you have any questions or inquiries, feel free to reach out:
      </p>
      <p className="contact-description">Phone no.:</p>
      <ul className="contact-info">
        <li>
          <strong>Pranav Kumar Jangam:</strong> +91 75690 89985
        </li>
        <li>
          <strong>Voore Risheel Kumar:</strong> +91 63041 99969
        </li>
        <li>
          <strong>Akshaya:</strong> +91 70327 96574
        </li>
        <li>
          <strong>Bheemireddy Charan Sai Reddy:</strong> +91 93815 79723
        </li>
        <li>
          <strong>Bannuru Charan Reddy:</strong> +91 82970 98020
        </li>
        <li>
          <strong>Mohammad Nida Madeeha:</strong> +91 63052 03512
        </li>
        <li>
          <strong>Kolipyaka Vyshnavi:</strong> +91 63020 74473
        </li>
      </ul>
    </div>
  );
}

export default Contact;
