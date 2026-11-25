/* Eklektik World app-shell cache. Live tiles and APIs are never cached. */
const CACHE = 'eklektik-world-v23-splash2';
const SHELL = [
  './',
  './index.html',
  './manifest.webmanifest',
  './apple-touch-icon.png',
  './a/icon0.png',
  './a/icon1.png',
  './a/icon192.png',
  './a/icon512_maskable.png',
  './a/maplibre.js',
  './a/liberty.js',
  './a/limb.js',
  './a/tr.js',
  './a/app.js',
  './a/poleN.png',
  './a/poleS.png',
  './a/gr.json',
  './a/ph.json'
];

self.addEventListener('install', function (event) {
  event.waitUntil((async function () {
    const cache = await caches.open(CACHE);
    await Promise.all(SHELL.map(async function (url) {
      try {
        const res = await fetch(url, { cache: 'reload' });
        if (res.ok) await cache.put(url, res);
      } catch (e) {}
    }));
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', function (event) {
  event.waitUntil((async function () {
    const keys = await caches.keys();
    await Promise.all(keys.filter(function (k) { return k !== CACHE; }).map(function (k) { return caches.delete(k); }));
    await self.clients.claim();
  })());
});

function isLiveData(url) {
  const s = url.href;
  return /\/tiles?\b|arcgisonline|openstreetmap|tile\.|mapbox|nominatim|photon|open-meteo|weather|geocod|wikimedia|wikipedia/i.test(s);
}

async function networkFirst(req) {
  try {
    const res = await fetch(req);
    if (res && res.ok && res.type === 'basic') {
      const cache = await caches.open(CACHE);
      cache.put(req, res.clone());
    }
    return res;
  } catch (e) {
    const hit = await caches.match(req);
    if (hit) return hit;
    if (req.mode === 'navigate') {
      const fallback = await caches.match('./index.html') || await caches.match('./');
      if (fallback) return fallback;
    }
    throw e;
  }
}

async function cacheFirst(req) {
  const hit = await caches.match(req);
  if (hit) return hit;
  const res = await fetch(req);
  if (res && res.ok && res.type === 'basic') {
    const cache = await caches.open(CACHE);
    cache.put(req, res.clone());
  }
  return res;
}

self.addEventListener('fetch', function (event) {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  if (isLiveData(url)) return;
  const dest = req.destination;
  const isDoc = dest === 'document' || url.pathname.endsWith('/') || url.pathname.endsWith('.html') || url.pathname.endsWith('sw.js');
  if (isDoc) {
    event.respondWith(networkFirst(req));
    return;
  }
  event.respondWith(cacheFirst(req));
});
