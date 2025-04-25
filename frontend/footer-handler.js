/**
 * Footer Handler - Manages the modern footer functionality
 */
document.addEventListener('DOMContentLoaded', function() {
  // Load the footer template
  loadFooter();
  
  // Set up event listeners and functionality once footer is loaded
  function setupFooter() {
    // Update current year for copyright
    const yearElement = document.querySelector('.current-year');
    if (yearElement) {
      yearElement.textContent = new Date().getFullYear();
    }
    
    // Back to top button functionality
    const backToTopButton = document.querySelector('.back-to-top');
    if (backToTopButton) {
      // Initially hide the button
      backToTopButton.style.opacity = '0';
      backToTopButton.style.visibility = 'hidden';
      
      // Show/hide back to top button based on scroll position
      window.addEventListener('scroll', function() {
        if (window.scrollY > 300) {
          backToTopButton.style.opacity = '1';
          backToTopButton.style.visibility = 'visible';
        } else {
          backToTopButton.style.opacity = '0';
          backToTopButton.style.visibility = 'hidden';
        }
      });
      
      // Smooth scroll to top
      backToTopButton.addEventListener('click', function(e) {
        e.preventDefault();
        window.scrollTo({
          top: 0,
          behavior: 'smooth'
        });
      });
    }
    
    // Style the back to top button
    styleBackToTop();
  }
  
  // Load footer template into all pages
  function loadFooter() {
    const footerPlaceholder = document.getElementById('footer-placeholder');
    if (footerPlaceholder) {
      fetch('footer-template.html')
        .then(response => response.text())
        .then(data => {
          footerPlaceholder.innerHTML = data;
          setupFooter();
        })
        .catch(error => {
          console.error('Error loading footer:', error);
        });
    }
  }
  
  // Add styles for back to top button
  function styleBackToTop() {
    const backToTopButton = document.querySelector('.back-to-top');
    if (backToTopButton) {
      backToTopButton.style.position = 'fixed';
      backToTopButton.style.bottom = '20px';
      backToTopButton.style.right = '20px';
      backToTopButton.style.width = '40px';
      backToTopButton.style.height = '40px';
      backToTopButton.style.backgroundColor = 'var(--primary)';
      backToTopButton.style.color = 'white';
      backToTopButton.style.borderRadius = '50%';
      backToTopButton.style.display = 'flex';
      backToTopButton.style.alignItems = 'center';
      backToTopButton.style.justifyContent = 'center';
      backToTopButton.style.textDecoration = 'none';
      backToTopButton.style.boxShadow = '0 2px 10px rgba(0, 98, 255, 0.2)';
      backToTopButton.style.zIndex = '999';
      backToTopButton.style.transition = 'all 0.3s ease';
      
      backToTopButton.addEventListener('mouseover', function() {
        this.style.transform = 'translateY(-5px)';
        this.style.boxShadow = '0 5px 15px rgba(0, 98, 255, 0.3)';
      });
      
      backToTopButton.addEventListener('mouseout', function() {
        this.style.transform = 'translateY(0)';
        this.style.boxShadow = '0 2px 10px rgba(0, 98, 255, 0.2)';
      });
    }
  }
});
