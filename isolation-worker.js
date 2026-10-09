// This service worker used to make the page cross-origin isolated, for threads the natural voices no longer use
// (they run on the GPU now). Browsers that installed it keep it until it's replaced, so this version removes
// itself and hands every open page back to the network.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', event => event.waitUntil(self.registration.unregister()));
