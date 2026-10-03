#!/usr/bin/env node
// Fills gaps in the HD recording from the camera's own SD card, through the TrueCloud cloud
// (the portal's "Download": search the card's recordings, replay a time range in download mode).
// The card records on its own, so it has what our recording lost when the camera's cloud link
// stalled or reconnected. Gaps newest first, one after another while the run lasts, each as a
// normal HD clip:
//   <REC_DIR>/<stream>/hd/<day>/<stream>-hd-<epoch>_<day>_<time>.mp4   (mtime = its last frame)
// and a line in <REC_DIR>/<stream>/hd/filled.log.
// Short live pieces (under SHORT_S of picture: what's left of a stretch where the cloud link
// struggled, often frozen or stretched) count as gaps too; once a card copy covers one whole,
// it's moved to <day>/replaced/ (the player ignores subfolders; they go with the day).
//
// The download shares the camera's upload with the live HD recording (and comes at ~0.2x real
// time), so it runs at quiet hours (sd-fill.timer: every 30 min, 01:00-06:00) and only when the
// live recording hasn't stalled or reconnected in the last 5 minutes; a run takes at most
// MAX_RUN_S and fetches at most MAX_CHUNK_S of recording, and stops if the download stalls.
//
//   node sd_fill.js DEVICE.json [stream=cam1] [--now] [--from=HH:MM --to=HH:MM [--day=YYYY-MM-DD]]
//   --now: ignore the quiet hours; --from/--to: only gaps in that window (local time, today
//   unless --day)
// Nothing secret is printed: the connector's own logging (which dumps its keys) is switched off.
const fs = require("fs"), path = require("path"), vm = require("vm");
const { spawn, execFileSync } = require("child_process");
const WebSocket = require("ws");
const { XMLHttpRequest } = require("xmlhttprequest-ssl");

const args = process.argv.slice(2).filter(a => !a.startsWith("--")), NOW = process.argv.includes("--now");
const opt = n => (process.argv.find(a => a.startsWith(`--${n}=`)) || "").split("=")[1];
const cfg = JSON.parse(fs.readFileSync(args[0], "utf8"));
const STREAM = args[1] || "cam1";
const REC_DIR = process.env.REC_DIR || "/home/ubuntu/rtmp-recordings";
const HD = path.join(REC_DIR, STREAM, "hd");
const TZ_S = 330 * 60;                  // the camera's times are local (India): epoch + 5:30
const KEEP_DAYS = 3;                    // as long as hd_record.py keeps HD
const SHORT_S = 180;
const MIN_GAP_S = 10, SETTLED_S = 15 * 60, MAX_CHUNK_S = 300, MAX_RUN_S = 25 * 60, STALL_S = 30;
const QUIET_HOURS = [1, 6];             // local hours [from, to)
const [pid, did, secret] = cfg.dev.split(":");
const began = Date.now();
// --from / --to: only this window (epoch seconds)
const WINDOW = (() => {
  if (!opt("from") || !opt("to")) return null;
  const day = opt("day") || new Date(Date.now() + 330 * 60e3).toISOString().slice(0, 10);
  const at = hhmm => Date.parse(`${day}T${hhmm}:00+05:30`) / 1000;
  return [at(opt("from")), at(opt("to"))];
})();

const log = (...a) => process.stderr.write(a.join(" ") + "\n");
console.log = console.error = console.warn = console.info = () => {};
const quit = (why, code = 0) => { log("sd_fill:", why); process.exit(code); };
const hm = s => new Date(s * 1000).toLocaleString("en-GB", { timeZone: "Asia/Kolkata", hour12: false });
const dayOf = s => new Date((s + TZ_S) * 1000).toISOString().slice(0, 10);
const clock = s => new Date((s + TZ_S) * 1000).toISOString().slice(11, 19).replace(/:/g, "-");

// ---- before anything: quiet hours, and the live recording not struggling
const hour = new Date(Date.now() + TZ_S * 1000).getUTCHours();
if (!NOW && !(QUIET_HOURS[0] <= hour && hour < QUIET_HOURS[1])) quit("not the quiet hours");
try {
  const recent = execFileSync("journalctl", ["-u", "hd-record", "--since", "-5min", "-o", "cat", "--no-pager"], { encoding: "utf8" });
  if (/video stopped|connect failed|disconnected/.test(recent)) quit("the live recording stalled lately: leaving the link alone");
} catch {}

// ---- what we have: the HD files' [start, end] (end = last write; hd_retime.py sets both from the
// camera's clock), finished ones only; with short=true, the short live pieces instead, and
// with "copies" the card copies
const filledNames = () => { try { return new Set(fs.readFileSync(path.join(HD, "filled.log"), "utf8").match(/\S+-hd-\d+_\S+\.mp4/g) || []); } catch { return new Set(); } };
function ourRanges(short = false) {
  const out = [], card = filledNames();
  for (const d of fs.readdirSync(HD).filter(x => /^\d{4}-\d\d-\d\d$/.test(x))) {
    for (const f of fs.readdirSync(path.join(HD, d))) {
      const m = f.match(/-hd-(\d+)_.*\.mp4$/); if (!m) continue;
      const end = fs.statSync(path.join(HD, d, f)).mtimeMs / 1000;
      if (end > Date.now() / 1000 - 30) continue;  // still being written
      const weak = !card.has(f) && end - m[1] < SHORT_S;
      if (short === "copies" ? card.has(f) : weak === short) out.push([+m[1], end, path.join(HD, d, f)]);
    }
  }
  return out.sort((a, b) => a[0] - b[0]);
}
// short live pieces the card copies (and `extra`, a copy not logged yet) cover whole: to <day>/replaced/
function replaceCovered(extra) {
  const gone = [], copies = merge([...ourRanges("copies").map(([a, b]) => [a, b]), ...(extra ? [extra] : [])]);
  for (const [a, b, f] of ourRanges(true)) {
    if (!copies.some(([c, d]) => c <= a + 1 && b - 1 <= d)) continue;
    fs.mkdirSync(path.join(path.dirname(f), "replaced"), { recursive: true });
    fs.renameSync(f, path.join(path.dirname(f), "replaced", path.basename(f)));
    gone.push(path.basename(f));
  }
  return gone;
}
const merge = r => [...r].sort((p, q) => p[0] - q[0]).reduce((o, [a, b]) => (o.length && a <= o.at(-1)[1] + 3 ? (o.at(-1)[1] = Math.max(o.at(-1)[1], b)) : o.push([a, b]), o), []);
// the parts of the card's ranges that ours don't cover
function gaps(card, ours) {
  const out = [], have = merge(ours);
  for (const [a, b] of merge(card)) {
    let t = a;
    for (const [c, d] of have) {
      if (d <= t || c >= b) continue;
      if (c > t) out.push([t, Math.min(c, b)]);
      t = Math.max(t, d);
    }
    if (t < b) out.push([t, b]);
  }
  const settled = Date.now() / 1000 - SETTLED_S;
  return out.map(([a, b]) => [a, Math.min(b, settled)]).filter(([a, b]) => b - a >= MIN_GAP_S);
}

// ---- MPEG-TS for ffmpeg (as in hd_source.js): H.265 + AAC ADTS, timestamps from the card's frames
const TS = (() => {
  const VPID = 0x100, APID = 0x101, PMT_PID = 0x1000, cc = {};
  const crc32 = b => { let c = 0xffffffff; for (const x of b) { c ^= x << 24; for (let i = 0; i < 8; i++) c = c & 0x80000000 ? ((c << 1) ^ 0x04c11db7) >>> 0 : (c << 1) >>> 0; } return c >>> 0; };
  const section = bytes => { const b = Buffer.from(bytes), c = Buffer.alloc(4); c.writeUInt32BE(crc32(b)); return Buffer.concat([Buffer.from([0]), b, c]); };
  const PAT = section([0x00, 0xb0, 13, 0x00, 0x01, 0xc1, 0x00, 0x00, 0x00, 0x01, 0xe0 | PMT_PID >> 8, PMT_PID & 0xff]);
  const PMT = section([0x02, 0xb0, 23, 0x00, 0x01, 0xc1, 0x00, 0x00, 0xe0 | VPID >> 8, VPID & 0xff, 0xf0, 0x00,
    0x24, 0xe0 | VPID >> 8, VPID & 0xff, 0xf0, 0x00, 0x0f, 0xe0 | APID >> 8, APID & 0xff, 0xf0, 0x00]);
  const pts5 = t => { const hi = Math.floor(t / 2 ** 30) % 8, mid = Math.floor(t / 2 ** 15) % 2 ** 15, lo = t % 2 ** 15;
    return [0x21 | hi << 1, mid >> 7, (mid & 0x7f) << 1 | 1, lo >> 7, (lo & 0x7f) << 1 | 1]; };
  const pcr6 = t => [Math.floor(t / 2 ** 25) % 256, Math.floor(t / 2 ** 17) % 256, Math.floor(t / 2 ** 9) % 256, Math.floor(t / 2) % 256, (t % 2) << 7 | 0x7e, 0];
  function packets(pid, payload, pcr = null) {
    const out = [];
    for (let off = 0, first = true; first || off < payload.length; first = false) {
      const pkt = Buffer.alloc(188, 0xff), left = payload.length - off;
      cc[pid] = ((cc[pid] ?? -1) + 1) & 15;
      let af = [];
      if (first && pcr != null) af = [0x10, ...pcr6(pcr)];
      const room = 184 - (af.length ? af.length + 1 : 0);
      if (left < room) { const need = room - left; if (!af.length) af = need === 1 ? null : [0x00, ...Array(need - 2).fill(0xff)]; else af.push(...Array(need).fill(0xff)); }
      pkt[0] = 0x47; pkt[1] = (first ? 0x40 : 0) | pid >> 8; pkt[2] = pid & 0xff;
      let h = 4;
      if (af === null) { pkt[3] = 0x30 | cc[pid]; pkt[4] = 0; h = 5; }
      else if (af.length) { pkt[3] = 0x30 | cc[pid]; pkt[4] = af.length; Buffer.from(af).copy(pkt, 5); h = 5 + af.length; }
      else pkt[3] = 0x10 | cc[pid];
      const n = Math.min(188 - h, left);
      payload.copy(pkt, h, off, off + n); off += n;
      out.push(pkt);
    }
    return out;
  }
  return {
    video(isKey, data, t) {
      const pes = Buffer.concat([Buffer.from([0, 0, 1, 0xe0, 0, 0, 0x80, 0x80, 5, ...pts5(t)]), Buffer.from(data)]);
      const pkts = packets(VPID, pes, t);
      return Buffer.concat(isKey ? [...packets(0, PAT), ...packets(PMT_PID, PMT), ...pkts] : pkts);
    },
    audio(data, t) {
      const d = Buffer.from(data), len = 3 + 5 + d.length;
      return Buffer.concat(packets(APID, Buffer.concat([Buffer.from([0, 0, 1, 0xc0, len >> 8, len & 0xff, 0x80, 0x80, 5, ...pts5(t)]), d])));
    },
  };
})();

// ---- the connection (TrueCloud's own connector, as hd_source.js)
const self = globalThis; self.self = self;
self.location = "https://www.cms.truecloudplus.co.in/static/433.connector.js";
self.WebSocket = WebSocket; self.XMLHttpRequest = XMLHttpRequest;
self.importScripts = (...u) => u.forEach(x => vm.runInThisContext(fs.readFileSync(path.join(__dirname, "truecloud", path.basename(x)), "utf8")));
const send = (message, data) => self.onmessage({ data: { message, data } });
const key = "f", chn = Array(128).fill(0); chn[0] = 1;

const today0 = Math.floor((Date.now() / 1000 + TZ_S) / 86400) * 86400 - TZ_S;  // local midnight, epoch
const days = Array.from({ length: KEEP_DAYS }, (_, i) => today0 - i * 86400);
let card = [], dayIdx = 0, gap = null, want = null;
let ff = null, frames = 0, firstTs = null, lastTs = null, lastAt = 0, startEpoch = null, tmp = null, out = null;

function searchNext() {
  if (dayIdx >= days.length) return chooseGap();
  const d0 = days[dayIdx++] + TZ_S;  // local seconds
  send("find_file_start_2", { sessionKey: key, chnlist: chn, begintime: d0, endtime: d0 + 86399, type: 15 });
}
// another gap while this run has time left and the live recording isn't struggling
function next() {
  if (Date.now() - began > MAX_RUN_S * 1000 - MAX_CHUNK_S * 1000) quit("this run's time is up");
  try {
    const recent = execFileSync("journalctl", ["-u", "hd-record", "--since", "-5min", "-o", "cat", "--no-pager"], { encoding: "utf8" });
    if (/video stopped|connect failed|disconnected/.test(recent)) quit("the live recording stalled: stopping for now");
  } catch {}
  send("replay_stop", { sessionKey: key });  // the last replay's stragglers mustn't reach the next clip
  want = null; lastAt = 0;
  setTimeout(() => { ff = null; frames = 0; firstTs = lastTs = startEpoch = tmp = out = null; finishing = false; chooseGap(); }, 1500);
}
function chooseGap() {
  const gone = replaceCovered();
  if (gone.length) { fs.appendFileSync(path.join(HD, "filled.log"), `${new Date().toISOString()} replaced ${gone.join(" ")} (covered by card copies)\n`); log(`sd_fill: moved ${gone.length} short live pieces the card copies cover to replaced/`); }
  const ours = ourRanges();
  log(`sd_fill: the card has ${card.length} recordings (${(card.reduce((s, [x, y]) => s + y - x, 0) / 3600).toFixed(1)} h), we have ${ours.length} HD files (${(ours.reduce((s, [x, y]) => s + y - x, 0) / 3600).toFixed(1)} h)`);
  let g = gaps(card, ours).sort((a, b) => b[0] - a[0]);
  if (WINDOW) g = g.map(([a, b]) => [Math.max(a, WINDOW[0]), Math.min(b, WINDOW[1])]).filter(([a, b]) => b - a >= MIN_GAP_S);
  const total = g.reduce((s, [a, b]) => s + b - a, 0);
  log(`sd_fill: ${g.length} gaps the card can fill, ${(total / 60).toFixed(1)} min in all`);
  if (!g.length) quit("nothing to fill");
  gap = g[0];
  want = [gap[0] - 1, Math.min(gap[1] + 1, gap[0] + MAX_CHUNK_S)];
  log(`sd_fill: filling ${hm(want[0])} - ${hm(want[1])}`);
  send("find_file_stop_2", { sessionKey: key });
  setTimeout(() => {
    lastAt = Date.now();
    send("replay_start", { sessionKey: key, channel: chn, starttime: want[0] + TZ_S, endtime: want[1] + TZ_S, type: 15, tasktype: 1 });
  }, 500);
}
// the card's frame times: local ms (epoch + 5:30) as far as seen; told apart from epoch ms by size
const tsEpoch = ts => { const s = ts / 1000; return Math.abs(s - TZ_S - want[0]) < Math.abs(s - want[0]) ? s - TZ_S : s; };
function onFrame(data) {
  if (!want || finishing) return;  // between gaps
  const isVideo = data.frametype === 1 || data.frametype === 2;
  if (!isVideo && data.frametype !== 0) return;
  if (!ff) {
    if (data.frametype !== 1) return;  // start on a key frame
    startEpoch = tsEpoch(data.ts_ms); firstTs = data.ts_ms;
    const day = dayOf(startEpoch);
    fs.mkdirSync(path.join(HD, day), { recursive: true });
    out = path.join(HD, day, `${STREAM}-hd-${Math.floor(startEpoch)}_${day}_${clock(startEpoch)}.mp4`);
    tmp = out + ".part";
    ff = spawn("ffmpeg", ["-nostdin", "-v", "error", "-y", "-f", "mpegts", "-i", "-", "-map", "0:v", "-map", "0:a?",
      "-c", "copy", "-tag:v", "hvc1", "-bsf:a", "aac_adtstoasc", "-movflags", "+faststart", "-f", "mp4", tmp], { stdio: ["pipe", "inherit", "inherit"] });
    ff.stdin.on("error", () => {});
  }
  lastAt = Date.now();
  const t = Math.round((data.ts_ms - firstTs) * 90) + 90000;
  if (isVideo) { frames++; lastTs = data.ts_ms; ff.stdin.write(TS.video(data.frametype === 1, data.frameData, t)); }
  else if (data.enc?.startsWith("AAC") && data.frameData[0] === 0xff && (data.frameData[1] & 0xf0) === 0xf0) ff.stdin.write(TS.audio(data.frameData, t));
  if (tsEpoch(data.ts_ms) >= want[1] - 1) finish("reached the end of the gap");
}
let finishing = false;
function finish(why) {
  if (finishing) return; finishing = true;
  if (!ff) quit(`${why}: nothing came`);
  const endEpoch = startEpoch + (lastTs - firstTs) / 1000;
  ff.on("close", code => {
    if (code !== 0 || !fs.existsSync(tmp)) { try { fs.unlinkSync(tmp); } catch {} quit(`${why}; ffmpeg failed (${code})`, 1); }
    fs.renameSync(tmp, out);
    fs.utimesSync(out, endEpoch, endEpoch);  // the player takes a clip's end from its mtime
    const gone = replaceCovered([startEpoch, endEpoch]);
    const line = `${new Date().toISOString()} filled ${hm(startEpoch)} - ${hm(endEpoch)} (${frames} frames, ${(endEpoch - startEpoch).toFixed(0)} s) ${path.basename(out)}; ${why}${gone.length ? `; replaced ${gone.join(" ")}` : ""}\n`;
    fs.appendFileSync(path.join(HD, "filled.log"), line);
    log(line.trim());
    next();
  });
  ff.stdin.end();
}

self.postMessage = ({ message, data }) => {
  if (message === "onconnect") { if (data.code !== 0) quit(`connect failed (${data.code})`, 1); send("login", { sessionKey: key, user: cfg.user, pwd: cfg.pwd, test: false }); }
  else if (message === "onloginresult") { if (data.code !== 0) quit(`login failed (${data.code})`, 1); searchNext(); }
  else if (message === "onsearchrec") card.push([data.file_begintime - TZ_S, data.file_endtime - TZ_S]);
  else if (message === "onsearchrecend") setTimeout(searchNext, 300);
  else if (message === "onrecvrecframe") onFrame(data);
  else if (message === "ondisconnect") { if (ff) finish("the connection dropped"); else quit("the connection dropped", 1); }
};
process.on("uncaughtException", e => quit(`error: ${e.message}`, 1));
importScripts("433.connector.js");
setTimeout(() => {
  send("create", { context: {}, sessionKey: key });
  send("connectbykey", { sessionKey: key, ngwDomain: cfg.ngw, ngwPort: cfg.port, productId: pid, deviceId: did, deviceSecret: secret, usehttps: cfg.https !== false, cb: false });
}, 500);
setInterval(() => {
  if (want && lastAt && Date.now() - lastAt > STALL_S * 1000) finish("the download stalled");
  if (Date.now() - began > MAX_RUN_S * 1000) { if (ff) finish("time's up for this run"); else quit("time's up", 1); }
}, 1000);
