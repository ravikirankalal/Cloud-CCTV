#!/usr/bin/env node
/*
 * Login check for nginx (auth_request). The page signs people in with Firebase (Google); this turns
 * that into a session cookie nginx can check on every request (pages' data, recordings, live streams).
 *
 *   POST /auth/session {idToken}  verify the Firebase ID token, set the session cookie
 *   GET  /auth/me                 200 {email} with a good cookie, else 401
 *   GET  /auth/check              what nginx asks: 204 or 401
 *   POST /auth/logout             clear the cookie
 *
 * Who may in: every Google account, or only the emails in ALLOWED (one per line) when that file has any.
 * Config (env): PORT 3300, FIREBASE_PROJECT, SECRET_FILE (random text, made on first start), ALLOWED.
 * No dependencies. Runs behind nginx on 127.0.0.1.
 */
const http = require('http'), crypto = require('crypto'), fs = require('fs'), path = require('path');
const PORT = +process.env.PORT || 3300;
const PROJECT = process.env.FIREBASE_PROJECT || 'localbc-41b52';
const DIR = process.env.AUTH_DIR || path.join(__dirname, 'state');
const SECRET_FILE = process.env.SECRET_FILE || path.join(DIR, 'auth-secret');
const ALLOWED = process.env.ALLOWED || path.join(DIR, 'allowed-emails.txt');
const JWKS = 'https://www.googleapis.com/service_accounts/v1/jwk/securetoken@system.gserviceaccount.com';
const DAYS = 7, COOKIE = 'nvr_session';

fs.mkdirSync(DIR, { recursive: true });
if (!fs.existsSync(SECRET_FILE)) fs.writeFileSync(SECRET_FILE, crypto.randomBytes(32).toString('hex'), { mode: 0o600 });
const SECRET = fs.readFileSync(SECRET_FILE, 'utf8').trim();

const b64u = b => Buffer.from(b).toString('base64url');
const sign = s => crypto.createHmac('sha256', SECRET).update(s).digest('base64url');
const same = (a, b) => a.length === b.length && crypto.timingSafeEqual(Buffer.from(a), Buffer.from(b));

let allowCache = { m: 0, list: [] };
function allowed(email) {
  try {
    const m = fs.statSync(ALLOWED).mtimeMs;
    if (m !== allowCache.m) allowCache = { m, list: fs.readFileSync(ALLOWED, 'utf8').split('\n').map(x => x.trim().toLowerCase()).filter(x => x && !x.startsWith('#')) };
  } catch { allowCache = { m: 0, list: [] }; }
  return !allowCache.list.length || allowCache.list.includes(email.toLowerCase());
}

let keys = { at: 0, keys: [] };
async function jwks() {
  if (Date.now() - keys.at < 3600e3 && keys.keys.length) return keys.keys;
  const r = await fetch(JWKS); if (!r.ok) throw new Error('jwks ' + r.status);
  keys = { at: Date.now(), keys: (await r.json()).keys }; return keys.keys;
}
// a Firebase ID token: RS256, signed by Google's key, for this project, not expired
async function verifyIdToken(tok) {
  const [h, p, s] = String(tok).split('.'); if (!s) throw new Error('format');
  const head = JSON.parse(Buffer.from(h, 'base64url')), pay = JSON.parse(Buffer.from(p, 'base64url'));
  if (head.alg !== 'RS256') throw new Error('alg');
  const jwk = (await jwks()).find(k => k.kid === head.kid); if (!jwk) throw new Error('kid');
  const ok = crypto.verify('RSA-SHA256', Buffer.from(h + '.' + p), crypto.createPublicKey({ key: jwk, format: 'jwk' }), Buffer.from(s, 'base64url'));
  const now = Date.now() / 1000;
  if (!ok || pay.aud !== PROJECT || pay.iss !== 'https://securetoken.google.com/' + PROJECT || !pay.sub || pay.exp < now || pay.iat > now + 60) throw new Error('claims');
  if (!pay.email || pay.email_verified === false) throw new Error('email');
  return pay;
}
function session(req) {
  const m = (req.headers.cookie || '').match(new RegExp('(?:^|; )' + COOKIE + '=([^;]+)'));
  if (!m) return null;
  const [p, sig] = m[1].split('.'); if (!sig || !same(sig, sign(p))) return null;
  try { const d = JSON.parse(Buffer.from(p, 'base64url')); return d.exp > Date.now() / 1000 && allowed(d.e) ? d : null; } catch { return null; }
}
const cookie = (v, age) => `${COOKIE}=${v}; Path=/; Max-Age=${age}; HttpOnly; Secure; SameSite=Lax`;
const send = (res, code, body, headers = {}) => { res.writeHead(code, { 'Cache-Control': 'no-store', ...headers }); res.end(body); };
const readBody = req => new Promise((ok, no) => { let b = ''; req.on('data', c => { b += c; if (b.length > 20000) { no(new Error('big')); req.destroy(); } }); req.on('end', () => ok(b)); });

http.createServer(async (req, res) => {
  const url = req.url.split('?')[0];
  try {
    if (url === '/auth/check') return send(res, session(req) ? 204 : 401);
    if (url === '/auth/me') { const d = session(req); return d ? send(res, 200, JSON.stringify({ email: d.e }), { 'Content-Type': 'application/json' }) : send(res, 401); }
    if (url === '/auth/logout' && req.method === 'POST') return send(res, 204, '', { 'Set-Cookie': cookie('', 0) });
    if (url === '/auth/session' && req.method === 'POST') {
      let pay;
      try { pay = await verifyIdToken(JSON.parse(await readBody(req)).idToken); }
      catch (e) { console.log('rejected token:', e.message); return send(res, 401, 'bad token'); }
      if (!allowed(pay.email)) { console.log('not allowed:', pay.email); return send(res, 403, 'not allowed'); }
      const p = b64u(JSON.stringify({ e: pay.email, exp: Math.floor(Date.now() / 1000) + DAYS * 86400 }));
      console.log('signed in:', pay.email);
      return send(res, 200, JSON.stringify({ email: pay.email }), { 'Content-Type': 'application/json', 'Set-Cookie': cookie(p + '.' + sign(p), DAYS * 86400) });
    }
    send(res, 404);
  } catch (e) { console.log('error', e); send(res, 500); }
}).listen(PORT, '127.0.0.1', () => console.log('auth_gate on', PORT));
