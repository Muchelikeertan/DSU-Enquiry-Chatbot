/**
 * DSU Official Homepage Interactivity Controller
 * Handles hero carousel, differentiators accordion, mobile navigation, and asset fallbacks.
 */

document.addEventListener("DOMContentLoaded", () => {
  initHeroCarousel();
  initDifferentiatorAccordion();
  initMobileNav();
  initAssetFallbacks();
  initBackToTop();
});

/**
 * 1. Hero Banner Carousel
 */
function initHeroCarousel() {
  const track = document.getElementById("hero-slider-track");
  const slides = document.querySelectorAll(".slide");
  const dots = document.querySelectorAll(".carousel-dot");
  const prevBtn = document.getElementById("hero-prev-btn");
  const nextBtn = document.getElementById("hero-next-btn");

  if (!track || slides.length === 0) return;

  let currentIndex = 0;
  const totalSlides = slides.length;
  let autoplayTimer = null;

  function goToSlide(index) {
    if (index < 0) {
      currentIndex = totalSlides - 1;
    } else if (index >= totalSlides) {
      currentIndex = 0;
    } else {
      currentIndex = index;
    }

    track.style.transform = `translateX(-${currentIndex * 100}%)`;

    dots.forEach((dot, idx) => {
      dot.classList.toggle("active", idx === currentIndex);
    });
  }

  function nextSlide() {
    goToSlide(currentIndex + 1);
  }

  function prevSlide() {
    goToSlide(currentIndex - 1);
  }

  function startAutoplay() {
    stopAutoplay();
    autoplayTimer = setInterval(nextSlide, 5000);
  }

  function stopAutoplay() {
    if (autoplayTimer) {
      clearInterval(autoplayTimer);
      autoplayTimer = null;
    }
  }

  if (nextBtn) {
    nextBtn.addEventListener("click", () => {
      nextSlide();
      startAutoplay();
    });
  }

  if (prevBtn) {
    prevBtn.addEventListener("click", () => {
      prevSlide();
      startAutoplay();
    });
  }

  dots.forEach((dot) => {
    dot.addEventListener("click", (e) => {
      const idx = parseInt(e.target.getAttribute("data-slide-index"), 10);
      if (!isNaN(idx)) {
        goToSlide(idx);
        startAutoplay();
      }
    });
  });

  const sliderContainer = document.querySelector(".hero-slider-section");
  if (sliderContainer) {
    sliderContainer.addEventListener("mouseenter", stopAutoplay);
    sliderContainer.addEventListener("mouseleave", startAutoplay);
  }

  startAutoplay();
}

/**
 * 2. DSU Differentiators Accordion
 */
function initDifferentiatorAccordion() {
  const accordionHeaders = document.querySelectorAll(".accordion-header");

  accordionHeaders.forEach((header) => {
    header.addEventListener("click", () => {
      const parentItem = header.parentElement;
      const isActive = parentItem.classList.contains("active");

      // Close all accordion items
      document.querySelectorAll(".accordion-item").forEach((item) => {
        item.classList.remove("active");
      });

      // Toggle clicked item
      if (!isActive) {
        parentItem.classList.add("active");
      }
    });
  });
}

/**
 * 3. Mobile Navigation Menu Toggle
 */
function initMobileNav() {
  const navToggle = document.getElementById("mobile-nav-toggle");
  const navMenu = document.getElementById("main-nav-menu");

  if (!navToggle || !navMenu) return;

  navToggle.addEventListener("click", () => {
    navMenu.classList.toggle("open");
  });
}

/**
 * 4. Graceful Asset Fallbacks
 * Catches any broken image links (e.g. if offline or if DSU CDN throttles)
 * and replaces with branded high-contrast neutral fallback elements.
 */
function initAssetFallbacks() {
  const allImages = document.querySelectorAll("img");

  allImages.forEach((img) => {
    img.addEventListener("error", function () {
      // If logo fails, show stylized text brand
      if (this.classList.contains("main-dsu-logo-img")) {
        this.style.display = "none";
        const fallback = document.getElementById("logo-text-fallback");
        if (fallback) fallback.style.display = "flex";
        return;
      }

      // If school or news image fails, set clean branded background
      this.style.opacity = "0";
      const parent = this.parentElement;
      if (parent) {
        parent.style.background = "linear-gradient(135deg, #003366 0%, #167ac6 100%)";
        parent.style.display = "flex";
        parent.style.alignItems = "center";
        parent.style.justifyContent = "center";
        parent.style.color = "#ffffff";
        parent.style.fontWeight = "bold";
      }
    });
  });
}

/**
 * 5. Back to Top Button
 */
function initBackToTop() {
  const backToTopBtn = document.getElementById("back-to-top-btn");
  if (!backToTopBtn) return;

  window.addEventListener("scroll", () => {
    if (window.scrollY > 300) {
      backToTopBtn.style.display = "block";
    } else {
      backToTopBtn.style.display = "none";
    }
  });

  backToTopBtn.addEventListener("click", (e) => {
    e.preventDefault();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
}
