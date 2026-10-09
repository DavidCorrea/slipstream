// Makes the page cross-origin isolated, which is what lets the browser run the natural voices (and anything else
// built on WebAssembly) on several threads: twice as fast for Kokoro, quick enough to keep up with a race.
// Isolation needs two headers on the site's own responses, and GitHub Pages can't send them, so this service
// worker adds them to every response from this site. `credentialless` lets other sites' files (the CDN, the
// fonts, the models) load as they did, just without cookies. A browser that doesn't support it ignores the
// headers and stays single-threaded; nothing else changes.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', event => event.waitUntil(self.clients.claim()));

self.addEventListener('fetch', event => {
  const { request } = event;
  // Other sites' responses can't be changed (and don't need to be); neither can this odd devtools request.
  if (new URL(request.url).origin !== self.location.origin) return;
  if (request.cache === 'only-if-cached' && request.mode !== 'same-origin') return;
  event.respondWith(fetch(request).then(response => {
    if (response.status === 0) return response;
    const headers = new Headers(response.headers);
    headers.set('Cross-Origin-Opener-Policy', 'same-origin');
    headers.set('Cross-Origin-Embedder-Policy', 'credentialless');
    return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
  }));
});
