/*
  Belo landing page behavior. Three small things, no dependencies:
  the attention toggle, the hold to teach demo, and a scroll reveal.
  Everything degrades to a working static page if this file never runs.
*/
(function () {
  "use strict";

  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* 1. Attention overlay toggle, mirroring the app's Show attention switch. */
  var toggle = document.querySelector("[data-attention-toggle]");
  var cameraSection = document.getElementById("camera");
  if (toggle && cameraSection) {
    toggle.addEventListener("click", function () {
      var on = cameraSection.getAttribute("data-attention") === "on";
      cameraSection.setAttribute("data-attention", on ? "off" : "on");
      toggle.setAttribute("aria-pressed", on ? "false" : "true");
    });
  }

  /* 2. Hold to teach. Same shape as the app: one frame per tap, a run of
        frames while held, capped at 60, at the selected capture rate. */
  var teach = document.querySelector("[data-teach]");
  var count = document.querySelector("[data-teach-count]");
  if (teach && count) {
    var MAX_PER_HOLD = 60;
    var RATE_HZ = 4;
    var timer = null;
    var taught = 0;

    function render() {
      count.textContent = taught === 0 ? "0" : String(taught);
    }

    function start(event) {
      event.preventDefault();
      if (timer !== null) return;
      taught = 0;
      teach.setAttribute("data-holding", "true");
      taught = 1;
      render();
      timer = window.setInterval(function () {
        if (taught >= MAX_PER_HOLD) {
          stop();
          return;
        }
        taught += 1;
        render();
      }, 1000 / RATE_HZ);
    }

    function stop() {
      if (timer !== null) {
        window.clearInterval(timer);
        timer = null;
      }
      teach.removeAttribute("data-holding");
    }

    teach.addEventListener("pointerdown", start);
    teach.addEventListener("pointerup", stop);
    teach.addEventListener("pointercancel", stop);
    teach.addEventListener("pointerleave", stop);
    teach.addEventListener("keydown", function (event) {
      if (event.key === " " || event.key === "Enter") start(event);
    });
    teach.addEventListener("keyup", stop);
    teach.addEventListener("blur", stop);
    render();
  }

  /* 3. Copy the clone block. */
  var copyButtons = document.querySelectorAll("[data-copy]");
  Array.prototype.forEach.call(copyButtons, function (button) {
    button.addEventListener("click", function () {
      var target = document.querySelector(button.getAttribute("data-copy"));
      if (!target || !navigator.clipboard) return;
      navigator.clipboard.writeText(target.textContent.trim()).then(function () {
        var previous = button.textContent;
        button.textContent = "Copied";
        window.setTimeout(function () {
          button.textContent = previous;
        }, 1400);
      });
    });
  });

  /* 4. Scroll reveal, motivated: sections arrive in reading order instead of
        all at once. Skipped entirely when the user asked for less motion. */
  if (reduced || !("IntersectionObserver" in window)) return;

  var targets = document.querySelectorAll(
    ".hero__copy, .hero__shot, .section__head, .flow__card, .camera__art, .camera__copy, .tile, .model__item, .projects__copy, .tree, .get__head, .card"
  );
  Array.prototype.forEach.call(targets, function (element) {
    element.classList.add("reveal");
  });

  var observer = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-in");
        observer.unobserve(entry.target);
      });
    },
    { rootMargin: "0px 0px -8% 0px", threshold: 0.12 }
  );

  Array.prototype.forEach.call(targets, function (element) {
    observer.observe(element);
  });

  /* Failsafe: a full page screenshot renders everything at once and never
     scrolls, so anything still hidden after a moment is revealed outright.
     Nothing on this page may depend on the observer to become readable. */
  window.setTimeout(function () {
    observer.disconnect();
    Array.prototype.forEach.call(targets, function (element) {
      element.classList.add("is-in");
    });
  }, 2000);
})();
