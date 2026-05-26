/* ════════════════════════════════════════════════════════════════════
   SARA — shared shell: theme persistence + nav rendering
   ════════════════════════════════════════════════════════════════════ */
(function(){
  'use strict';

  const THEME_KEY = 'sara-theme';
  const DEFAULT_THEME = 'dark';

  // Resolve initial theme as early as possible to avoid flash.
  // (This script is loaded with `defer`, so DOMContentLoaded has not fired yet
  //  but <html> already exists in the parser.)
  const savedTheme = (function(){
    try { return localStorage.getItem(THEME_KEY); } catch(e){ return null; }
  })();
  const initialTheme = savedTheme === 'light' || savedTheme === 'dark'
    ? savedTheme
    : DEFAULT_THEME;
  document.documentElement.setAttribute('data-theme', initialTheme);

  // ── public API ─────────────────────────────────────────────────────
  window.SARA = window.SARA || {};

  window.SARA.getTheme = function(){
    return document.documentElement.getAttribute('data-theme') || DEFAULT_THEME;
  };

  window.SARA.setTheme = function(theme){
    if (theme !== 'light' && theme !== 'dark') return;
    document.documentElement.setAttribute('data-theme', theme);
    try { localStorage.setItem(THEME_KEY, theme); } catch(e){}
    // Notify listeners (e.g. Plotly chart needs to re-style)
    window.dispatchEvent(new CustomEvent('sara:themechange', { detail: { theme } }));
  };

  window.SARA.toggleTheme = function(){
    window.SARA.setTheme(window.SARA.getTheme() === 'dark' ? 'light' : 'dark');
  };

  // ── nav renderer ───────────────────────────────────────────────────
  // Usage: <div data-sara-nav data-page="authors" data-meta="115 authors"></div>
  window.SARA.renderNav = function(opts){
    opts = opts || {};
    const host = opts.host || document.querySelector('[data-sara-nav]');
    if (!host) return;

    const page = opts.page || host.dataset.page || '';
    const meta = opts.meta || host.dataset.meta || '';

    const links = [
      { id: 'home',         label: 'Home',         href: 'index.html' },
      { id: 'authors',      label: 'Authors Map',  href: 'authors.html' },
      { id: 'publications', label: 'Publications', href: 'publications.html' },
    ];

    const linksHtml = links.map(function(l){
      const active = l.id === page ? ' active' : '';
      return '<a class="sara-nav__link' + active + '" href="' + l.href + '">' + l.label + '</a>';
    }).join('');

    host.className = 'sara-nav';
    host.innerHTML =
      '<div class="sara-nav__brand">' +
        '<div class="sara-nav__brand-dot"></div>' +
        '<span>SARA · WMiI UAM</span>' +
      '</div>' +
      '<nav class="sara-nav__links">' + linksHtml + '</nav>' +
      '<div class="sara-nav__spacer"></div>' +
      '<div class="sara-nav__meta" id="sara-nav-meta">' + (meta || '') + '</div>' +
      '<button class="sara-theme-toggle" type="button" title="Toggle theme" aria-label="Toggle theme">' +
        '<span class="sara-theme-toggle__moon">◐</span>' +
        '<span class="sara-theme-toggle__sun">☀</span>' +
      '</button>';

    host.querySelector('.sara-theme-toggle').addEventListener('click', window.SARA.toggleTheme);
  };

  window.SARA.setNavMeta = function(text){
    const el = document.getElementById('sara-nav-meta');
    if (el) el.textContent = text || '';
  };

  // Auto-render once DOM is ready.
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function(){ window.SARA.renderNav(); });
  } else {
    window.SARA.renderNav();
  }
})();
