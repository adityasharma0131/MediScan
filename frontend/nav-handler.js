document.addEventListener('DOMContentLoaded', function() {
  // Load navigation template
  const navPlaceholder = document.getElementById('nav-placeholder');
  if (navPlaceholder) {
    fetch('nav-template.html')
      .then(response => response.text())
      .then(data => {
        navPlaceholder.innerHTML = data;
        
        // Set active navigation link based on current page
        const currentPage = window.location.pathname.split('/').pop();
        setActiveNavLink(currentPage);
        
        // Initialize mobile menu functionality
        initializeMobileNav();
      })
      .catch(error => console.error('Error loading navigation:', error));
  }
  
  // Set the active navigation link
  function setActiveNavLink(currentPage) {
    if (!currentPage) currentPage = 'index.html'; // Default to home page
    
    // Get all navigation links (both desktop and mobile)
    const desktopLinks = document.querySelectorAll('.nav-links a');
    const mobileLinks = document.querySelectorAll('.mobile-nav-links a');
    
    // Function to highlight the active link
    const highlightLink = (links) => {
      links.forEach(link => {
        if (link.getAttribute('href') === currentPage) {
          link.style.color = 'var(--primary)';
          // Add an underline effect
          link.style.fontWeight = '700';
          
          if (link.parentElement.classList.contains('nav-links')) {
            // For desktop navigation
            link.style.position = 'relative';
            link.innerHTML += '<span class="active-indicator"></span>';
          }
        }
      });
    };
    
    // Apply to both navigation menus
    highlightLink(desktopLinks);
    highlightLink(mobileLinks);
  }
  
  // Initialize mobile navigation functionality
  function initializeMobileNav() {
    const mobileMenuButton = document.querySelector('.mobile-menu-button');
    const mobileNavClose = document.querySelector('.mobile-nav-close');
    const mobileNav = document.querySelector('.mobile-nav');
    
    if (mobileMenuButton) {
      mobileMenuButton.addEventListener('click', () => {
        mobileNav.classList.add('active');
        document.body.style.overflow = 'hidden';
      });
    }
    
    if (mobileNavClose) {
      mobileNavClose.addEventListener('click', () => {
        mobileNav.classList.remove('active');
        document.body.style.overflow = '';
      });
    }
    
    // Close mobile nav when clicking a link
    const mobileNavLinks = document.querySelectorAll('.mobile-nav-links a');
    if (mobileNavLinks.length > 0) {
      mobileNavLinks.forEach(link => {
        link.addEventListener('click', () => {
          mobileNav.classList.remove('active');
          document.body.style.overflow = '';
        });
      });
    }
    
    // Header scroll effect
    window.addEventListener('scroll', () => {
      const header = document.getElementById('header');
      if (window.scrollY > 50) {
        header.classList.add('scrolled');
      } else {
        header.classList.remove('scrolled');
      }
    });
  }
});