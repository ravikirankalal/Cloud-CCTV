#!/usr/bin/env node
// One camera's HD stream from the TrueCloud cloud, as MPEG-TS on stdout (recorder.py runs it).
// Runs the TrueCloud portal's own p2p worker (truecloud/*.connector.js, the dvr163 "kp2p"
// protocol): find the device via the vendor's servers, connect through their relay, log in
// with the device key, open the stream. Exits (non-zero) when the link fails or goes quiet;
// recorder.py starts it again.
//
//   node camera_stream.js DEVICE.json [streamid=1]
// DEVICE.json: {"dev": "productId:deviceId:deviceSecret", "user", "pwd", "ngw", "port", "https"}
// (saved from the portal, see experiments/truecloud/grab-device.js). Stream 1 = HD (2304x1296
// H.265 on the Common camera), 0 = SD (800x448 H.264, what RTMP sends).
// Nothing secret is ever printed: the worker's own logging (which dumps its connection,
// key included) is switched off; messages go to stderr.
//
// Live view with no delay (HLS waits for whole pieces, ~10 s each with this camera's
// keyframes): with HD_LIVE_SOCK (a Unix socket; nginx proxies /api/hd-live/<stream> to it) or
// HD_LIVE_PORT (a local port, for testing) set, the frames also go straight to the browsers
// watching, as fragmented MP4 over a WebSocket (see "the live relay" below).
// For the recorder, picture and sound go out on stdout together as one MPEG-TS stream with
// timestamps set here (see "the recording stream" below): ffmpeg reading them as two inputs
// held the picture back by minutes to keep them in step.
// A new viewer first gets the frames since the last key frame, so it can start at once.
const fs = require("fs"), path = require("path"), vm = require("vm");
const WebSocket = require("ws");
const { XMLHttpRequest } = require("xmlhttprequest-ssl");

const JS = path.join(__dirname, "truecloud");
const cfg = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const STREAM = +(process.argv[3] || 1);
const CONNECT_S = 40;   // give up if no video this long after starting
const REREQUEST_S = 6;  // no video this long: ask for the stream again
const QUIET_S = 20;     // ... or if video stops this long (a slow link stalls for seconds: wait it out;
                        // reconnecting costs 20-30 s, and the 10-min cut-off is renewed away)
const [pid, did, secret] = cfg.dev.split(":");

const log = (...a) => process.stderr.write(a.join(" ") + "\n");
console.log = console.error = console.warn = console.info = () => {};
const quit = (why, code = 1) => { log("camera_stream:", why); process.exit(code); };
// Cloud/device "not available right now" codes: -15 rate-limit after a drop (~50 s window),
// -13 refused, -12 device seek timeout (camera offline/rebooting). We exit 2 on these so the
// recorder backs off past the refusal window instead of hammering it; a clean drop stays exit 1.
const REFUSAL = new Set([-15, -13, -12]);
const REFUSED_EXIT = 2;

const self = globalThis;
self.self = self;
self.location = "https://www.cms.truecloudplus.co.in/static/433.connector.js";
self.WebSocket = WebSocket;
self.XMLHttpRequest = XMLHttpRequest;
self.importScripts = (...urls) => urls.forEach(u =>
  vm.runInThisContext(fs.readFileSync(path.join(JS, path.basename(u)), "utf8"), { filename: u }));
const send = (message, data) => self.onmessage({ data: { message, data } });

const key = "s";
const RENEW_S = +(process.env.RENEW_S ?? 0);  // re-send open_stream this often; 0 = off (default).
// Off on purpose: never disturb a working stream. The 4-min renewal does NOT prevent this camera's
// stalls (it stalls its video + SD-card recording ~every 4-5 min regardless — a card/encoder issue,
// not our link), and re-sending open_stream mid-stream can itself coincide with a teardown. So we hold
// the session and only reconnect when it actually drops.
let renew = null;
// once a minute: frames and bytes received, and how far the camera's frame clock has fallen
// behind ours since the start (the cloud relay delivers late when it can't keep up)
const stats = { n: 0, bytes: 0, ts: null, ts0: null, wall0: null, sound: 0, silent: 0, dropped: 0 };
setInterval(() => {
  if (stats.ts0 == null) return;
  const behind = ((Date.now() - stats.wall0) - (stats.ts - stats.ts0)) / 1000;
  log(`camera_stream: ${stats.n} frames, ${stats.sound} sound (+${stats.silent} silent, -${stats.dropped} dropped), ` +
      `${(stats.bytes * 8 / 60e3).toFixed(0)} kbit/s, ${behind.toFixed(1)} s behind the camera; longest stall ${(loopDelay.max / 1e6).toFixed(0)} ms; ` +
      `recorder queue ${(videoOut.writableLength / 1e6).toFixed(1)} MB, dropped ${outs.video.dropped}; ` +
      `relay queue ${mux ? (mux.stdin.writableLength / 1e6).toFixed(1) : 0} MB`);
  loopDelay.reset(); outs.video.dropped = 0;
  stats.n = 0; stats.bytes = 0; stats.sound = stats.silent = stats.dropped = 0;
}, 60000);
let started = false, last = Date.now(), rerequested = false;

// ---- the live relay: frames into fragmented MP4 (a small ffmpeg, no re-encoding: 200 ms
// fragments), sent to the browsers watching over a WebSocket, which play it with Media Source
// Extensions (they work on plain http; WebCodecs needs https). Messages: first a text one,
// {"t0": epoch s of video media time 0}, then binary ones whose first byte says what follows:
// 0 = video (the init segment, then fragments), 1 = sound (likewise). A new viewer gets the
// video fragments since the last key frame, so it starts at once; sound from then on.
// Picture and sound are two separate ffmpegs and two tracks: one ffmpeg with both inputs held
// the picture back by minutes to keep them in step (the sound trickles in), and in the browser
// they play in separate elements, so late sound never holds the picture either.
const viewers = new Set();
const MAX_BEHIND = 4e6;  // bytes queued for a slow viewer before it skips to the next key frame
let mux = null, init = null, hello = null, gop = [];
let amux = null, ainit = null;  // the sound's ffmpeg and init segment
const tag = (kind, buf) => Buffer.concat([Buffer.from([kind]), buf]);
const keyQueue = [];     // key flags of the video frames sent to the muxer, not yet out of it
function boxes(b, from = 0, to = b.length) {  // top-level MP4 boxes in b[from:to]: [{type, start, end}]
  const out = [];
  for (let i = from; i + 8 <= to;) {
    const size = b.readUInt32BE(i); if (size < 8 || i + size > to) break;
    out.push({ type: b.toString("latin1", i + 4, i + 8), start: i, end: i + size });
    i += size;
  }
  return out;
}
// the number of video samples in a fragment's moof (video is track 1), and so its key flags
function videoSamples(moof) {
  let n = 0;
  for (const traf of boxes(moof, 8).filter(x => x.type === "traf")) {
    const kids = boxes(moof, traf.start + 8, traf.end);
    const tfhd = kids.find(x => x.type === "tfhd"), trun = kids.find(x => x.type === "trun");
    if (tfhd && trun && moof.readUInt32BE(tfhd.start + 12) === 1) n += moof.readUInt32BE(trun.start + 12);
  }
  return n;
}
function sendTo(ws, msg) {
  if (ws.bufferedAmount > MAX_BEHIND) { ws.skipping = true; return; }
  ws.send(msg);
}
function greet(ws) {
  ws.send(hello); ws.send(init);
  if (ainit) ws.send(ainit);
  for (const f of gop) ws.send(f);
  ws.ready = true;
}
function fragment(frag, isKey) {
  if (isKey) gop = [];
  else if (!gop.length) return;  // nothing before the first key frame
  gop.push(frag);
  for (const ws of viewers) {
    if (!ws.ready) continue;
    if (ws.skipping) { if (!isKey) continue; ws.skipping = false; }
    sendTo(ws, frag);
  }
}
function liveStart() {
  const { spawn } = require("child_process");
  // video only: with the sound as a second input, ffmpeg held the picture back to keep the
  // two in step (the sound trickles in), and live fell minutes behind
  mux = spawn("ffmpeg", ["-nostdin", "-v", "error",
    "-use_wallclock_as_timestamps", "1", "-f", "hevc", "-i", "pipe:0",
    "-map", "0:v", "-c:v", "copy", "-tag:v", "hvc1",
    // frames come in bursts from the cloud: arrival times a few microseconds apart, which MSE
    // won't take; space them out (60 ms), else keep arrival time
    "-bsf:v", "setts=ts=if(isnan(PREV_OUTPTS)\\,PTS\\,max(PTS\\,PREV_OUTPTS+0.06/TB))",
    // a fragment every 200 ms, and one at each key frame (where a new viewer starts); a fragment
    // per frame doesn't play in Chrome's MSE
    "-f", "mp4", "-frag_duration", "200000", "-movflags", "frag_keyframe+empty_moov+default_base_moof",
    "-flush_packets", "1", "pipe:1"],
    { stdio: ["pipe", "pipe", "inherit"] });
  mux.on("exit", () => quit("live muxer stopped"));
  for (const st of [mux.stdin, mux.stdout]) st.on("error", () => {});
  let buf = Buffer.alloc(0), head = [];
  mux.stdout.on("data", d => {
    buf = buf.length ? Buffer.concat([buf, d]) : d;
    let used = 0;
    for (const x of boxes(buf)) {
      const box = buf.subarray(x.start, x.end);
      used = x.end;
      if (x.type === "ftyp" || x.type === "moof") head = [Buffer.from(box)];
      else if (x.type === "moov") {
        init = tag(0, Buffer.concat([...head, box])); head = [];
        hello = JSON.stringify({ t0: mux.t0 });
        for (const ws of viewers) if (!ws.ready) greet(ws);
      } else if (x.type === "mdat" && head.length) {
        const moof = head[0], n = videoSamples(moof);
        const keys = keyQueue.splice(0, n);
        fragment(tag(0, Buffer.concat([moof, box])), n > 0 && keys[0] === true);
        head = [];
      }
    }
    buf = Buffer.from(buf.subarray(used));
  });
}
function liveFrame(isKey, data) {
  if (!mux) return;
  if (mux.t0 == null) mux.t0 = Date.now() / 1000;  // media time 0 (ffmpeg starts the first frame at 0)
  keyQueue.push(isKey);
  mux.stdin.write(Buffer.from(data));
}
// the sound: its own ffmpeg (re-encoded, 8 kHz mono: next to no work; copied, its setup isn't
// known when the header is written, and MSE won't take a header without it)
function audioStart() {
  const { spawn } = require("child_process");
  const a = amux = spawn("ffmpeg", ["-nostdin", "-v", "error",
    "-f", "aac", "-i", "pipe:0",   // timed by the frames themselves: arrival times jitter, and
    "-c:a", "aac", "-b:a", "24k",  // resampling to them put a gap in every piece of sound
    "-f", "mp4", "-frag_duration", "200000", "-movflags", "empty_moov+default_base_moof",
    "-flush_packets", "1", "pipe:1"], { stdio: ["pipe", "pipe", "inherit"] });
  for (const st of [a.stdin, a.stdout]) st.on("error", () => {});
  a.on("exit", () => { if (amux === a) { amux = null; ainit = null; log("camera_stream: live sound stopped"); } });  // the picture carries on
  let buf = Buffer.alloc(0), head = [];
  a.stdout.on("data", d => {
    buf = buf.length ? Buffer.concat([buf, d]) : d;
    let used = 0;
    for (const x of boxes(buf)) {
      const box = buf.subarray(x.start, x.end);
      used = x.end;
      if (x.type === "ftyp" || x.type === "moof") head = [Buffer.from(box)];
      else if (x.type === "moov") {
        ainit = tag(1, Buffer.concat([...head, box])); head = [];
        for (const ws of viewers) if (ws.ready) ws.send(ainit);
      } else if (x.type === "mdat" && head.length) {
        const frag = tag(1, Buffer.concat([head[0], box])); head = [];
        for (const ws of viewers) if (ws.ready && !ws.skipping && ws.bufferedAmount <= MAX_BEHIND) ws.send(frag);
      }
    }
    buf = Buffer.from(buf.subarray(used));
  });
}
function liveAudio(data) { if (amux && mux?.t0 != null) amux.stdin.write(Buffer.from(data)); }
const LIVE = !!(process.env.HD_LIVE_SOCK || process.env.HD_LIVE_PORT);
if (LIVE) {
  const http = require("http");
  const server = http.createServer((req, res) => { res.writeHead(426); res.end(); });
  const wss = new WebSocket.Server({ server, perMessageDeflate: false });
  wss.on("connection", ws => {
    viewers.add(ws);
    if (init) greet(ws);
    ws.on("close", () => viewers.delete(ws));
    ws.on("error", () => viewers.delete(ws));
  });
  const sock = process.env.HD_LIVE_SOCK;
  if (sock) {
    try { fs.unlinkSync(sock); } catch {}
    server.listen(sock, () => fs.chmodSync(sock, 0o666));  // nginx in the container runs as another user
  } else server.listen(+process.env.HD_LIVE_PORT, "127.0.0.1");
}
// Never let a consumer hold up reading from the camera: writes to a pipe from process.stdout
// block on Linux, and while they do, frames back up in the camera's cloud (they then arrive
// minutes late). So: non-blocking writes, and a consumer that falls far behind loses frames
// (up to the next key frame) instead of slowing the source down.
const videoOut = fs.createWriteStream(null, { fd: 1 });
videoOut.on("error", () => quit("output closed", 0));
const BEHIND_MAX = 16e6;  // bytes queued for a consumer before it skips to the next key frame
const outs = { video: { s: videoOut, skip: false, dropped: 0 } };
function feed(o, buf, isKey) {
  if (o.skip && !isKey) { o.dropped++; return; }
  if (o.s.writableLength > BEHIND_MAX) { o.skip = true; o.dropped++; return; }
  o.skip = false;
  o.s.write(buf);
}
const { monitorEventLoopDelay } = require("perf_hooks");
const loopDelay = monitorEventLoopDelay({ resolution: 20 }); loopDelay.enable();
// ---- sound: laid end to end on its own sample clock (1024 samples per AAC frame), the way MP4
// stores it anyway: where the camera skipped frames (a jump in its frame clock), silent frames
// fill the hole, and if the sound still drifts more than 1 s from the picture's clock (arrival
// times), silence brings it back level, or frames are dropped while it's ahead. Without this a
// lost stretch of sound slid everything after it earlier, by up to minutes.
const sound = { pts: null, ts: null, sil: undefined, step: 0, ms: 0 };
function adtsInfo(b) {
  const rates = [96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350];
  return { rate: rates[(b[2] >> 2) & 15] || 8000, ch: ((b[2] & 1) << 2) | (b[3] >> 6) || 1 };
}
function silentFrame(b) {  // one silent AAC frame (ADTS) in the camera's format, made by ffmpeg once
  const { rate, ch } = adtsInfo(b);
  try {
    const r = require("child_process").spawnSync("ffmpeg", ["-nostdin", "-v", "error", "-f", "lavfi", "-i",
      `anullsrc=r=${rate}:cl=${ch > 1 ? "stereo" : "mono"}`, "-t", "1", "-c:a", "aac", "-b:a", "24k", "-f", "adts", "pipe:1"]);
    const frames = [];
    for (let i = 0; i + 7 <= r.stdout.length;) { const n = ((r.stdout[i + 3] & 3) << 11) | (r.stdout[i + 4] << 3) | (r.stdout[i + 5] >> 5); if (n < 7) break; frames.push(r.stdout.subarray(i, i + n)); i += n; }
    return frames.length > 4 ? Buffer.from(frames[4]) : null;  // past the encoder's start-up frames
  } catch { return null; }
}
function soundFrames(data, camTs) {  // -> [[ADTS frame, 90 kHz time], ...]
  const d = Buffer.from(data), wall = TS.now(), out = [];
  if (wall == null || d.length < 7) return out;
  if (sound.sil === undefined) {
    sound.sil = silentFrame(d);
    const { rate } = adtsInfo(d); sound.step = 1024 / rate * 90000; sound.ms = 1024 / rate * 1000;
  }
  let fill = 0;
  if (sound.pts != null) {
    if (sound.ts != null && camTs > sound.ts) fill = Math.round((camTs - sound.ts) / sound.ms) - 1;  // frames the camera skipped
    if (fill > 300) fill = 0;                                                            // a clock jump, not a gap
    const ahead = sound.pts + (fill + 1) * sound.step - wall;
    if (ahead < -90000) fill += Math.floor(-ahead / sound.step);                         // > 1 s behind: catch up
    else if (ahead > 90000 && fill === 0) { sound.ts = camTs; stats.dropped++; return out; } // > 1 s ahead: drop
  }
  sound.ts = camTs;
  if (sound.pts == null) sound.pts = wall - sound.step;
  for (let i = 0; i < Math.min(fill, 1000) && sound.sil; i++) { out.push([sound.sil, Math.round(sound.pts += sound.step)]); stats.silent++; }
  out.push([d, Math.round(sound.pts += sound.step)]); stats.sound++;
  return out;
}

// ---- the recording stream: MPEG-TS, H.265 (PID 0x100) and AAC ADTS (PID 0x101). Timestamps
// are arrival times on one clock (so sound stays with the picture), spaced at least a frame
// apart: frames come from the cloud in bursts.
function makeTS() {  // an MPEG-TS writer; TS (below) is the live recording's, a replay gets its own
  const VPID = 0x100, APID = 0x101, PMT_PID = 0x1000, cc = {};
  const crc32 = b => {
    let c = 0xffffffff;
    for (const x of b) { c ^= x << 24; for (let i = 0; i < 8; i++) c = c & 0x80000000 ? ((c << 1) ^ 0x04c11db7) >>> 0 : (c << 1) >>> 0; }
    return c >>> 0;
  };
  const section = bytes => { const b = Buffer.from(bytes), c = Buffer.alloc(4); c.writeUInt32BE(crc32(b)); return Buffer.concat([Buffer.from([0]), b, c]); };
  const PAT = section([0x00, 0xb0, 13, 0x00, 0x01, 0xc1, 0x00, 0x00, 0x00, 0x01, 0xe0 | PMT_PID >> 8, PMT_PID & 0xff]);
  // video stream_type: 0x24 = H.265/HEVC (the HD main stream + SD-card replay), 0x1b = H.264/AVC (the
  // SD sub-stream, streamid 0). The cloud reports the codec on the first frame (data.enc); setCodec()
  // rebuilds the PMT before it is first emitted (on that first key frame), so ffmpeg sees the right codec.
  const buildPMT = vtype => section([0x02, 0xb0, 23, 0x00, 0x01, 0xc1, 0x00, 0x00, 0xe0 | VPID >> 8, VPID & 0xff, 0xf0, 0x00,
    vtype, 0xe0 | VPID >> 8, VPID & 0xff, 0xf0, 0x00,
    0x0f, 0xe0 | APID >> 8, APID & 0xff, 0xf0, 0x00]);   // AAC (ADTS)
  let PMT = buildPMT(0x24);
  // 33-bit 90 kHz time in the PES / PCR layouts (no 32-bit bit operations on it)
  const pts5 = t => { const hi = Math.floor(t / 2 ** 30) % 8, mid = Math.floor(t / 2 ** 15) % 2 ** 15, lo = t % 2 ** 15;
    return [0x21 | hi << 1, mid >> 7, (mid & 0x7f) << 1 | 1, lo >> 7, (lo & 0x7f) << 1 | 1]; };
  const pcr6 = t => [Math.floor(t / 2 ** 25) % 256, Math.floor(t / 2 ** 17) % 256, Math.floor(t / 2 ** 9) % 256, Math.floor(t / 2) % 256, (t % 2) << 7 | 0x7e, 0];
  function packets(pid, payload, pcr = null) {
    const out = [];
    for (let off = 0, first = true; first || off < payload.length; first = false) {
      const pkt = Buffer.alloc(188, 0xff), left = payload.length - off;
      cc[pid] = ((cc[pid] ?? -1) + 1) & 15;
      let af = [];
      if (first && pcr != null) af = [0x10, ...pcr6(pcr)];          // flags: PCR
      const room = 184 - (af.length ? af.length + 1 : 0);
      if (left < room) {                                             // stuffing
        const need = room - left;
        if (!af.length) af = need === 1 ? null : [0x00, ...Array(need - 2).fill(0xff)];
        else af.push(...Array(need).fill(0xff));
      }
      pkt[0] = 0x47; pkt[1] = (first ? 0x40 : 0) | pid >> 8; pkt[2] = pid & 0xff;
      let h = 4;
      if (af === null) { pkt[3] = 0x30 | cc[pid]; pkt[4] = 0; h = 5; }  // one byte: an empty adaptation field
      else if (af.length) { pkt[3] = 0x30 | cc[pid]; pkt[4] = af.length; Buffer.from(af).copy(pkt, 5); h = 5 + af.length; }
      else pkt[3] = 0x10 | cc[pid];
      const n = Math.min(188 - h, left);
      payload.copy(pkt, h, off, off + n); off += n;
      out.push(pkt);
    }
    return out;
  }
  let t0 = null;
  const last = { v: -Infinity, a: -Infinity };
  const STEP = { v: 0.06 * 90000, a: 1024 / 8000 * 90000 };  // the least spacing: 60 ms, an AAC frame
  function stamp(kind) {
    if (t0 == null) {
      t0 = Date.now();
      // the wall-clock time of the first frame (a key frame: the first segment starts with it);
      // the recorder times segments from it (ffmpeg's own PROGRAM-DATE-TIME comes ~5 s late,
      // after it has probed the input)
      if (process.env.REC_T0_FILE) fs.writeFileSync(process.env.REC_T0_FILE, String(t0 / 1000));
    }
    const t = Math.max(Math.round((Date.now() - t0) * 90) + 90000, Math.round(last[kind] + STEP[kind]));
    last[kind] = t;
    return t;
  }
  return {
    video(isKey, data, t = stamp("v")) {
      const d = Buffer.from(data);
      const pes = Buffer.concat([Buffer.from([0, 0, 1, 0xe0, 0, 0, 0x80, 0x80, 5, ...pts5(t)]), d]);  // length 0: video may be unbounded
      const pkts = packets(VPID, pes, t);
      return Buffer.concat(isKey ? [...packets(0, PAT), ...packets(PMT_PID, PMT), ...pkts] : pkts);
    },
    now: () => t0 == null ? null : Math.round((Date.now() - t0) * 90) + 90000,
    setCodec(enc) { PMT = buildPMT(enc && enc.startsWith("H264") ? 0x1b : 0x24); },
    audio(d, t) {
      const len = 3 + 5 + d.length;
      const pes = Buffer.concat([Buffer.from([0, 0, 1, 0xc0, len >> 8, len & 0xff, 0x80, 0x80, 5, ...pts5(t)]), d]);
      return Buffer.concat(packets(APID, pes));
    },
  };
}
const TS = makeTS();
self.postMessage = ({ message, data }) => {
  if (message === "onrecvframeex") {
    if (data.frametype === 0) {                                      // audio (AAC, ADTS)
      if (!started || !data.enc.startsWith("AAC")) return;           // from the first video frame
      for (const [d, t] of soundFrames(data.frameData, data.timestamp)) {
        feed(outs.video, TS.audio(d, t), false);
        if (d !== sound.sil) liveAudio(d);  // live: real sound only (it plays as it comes; the silence is for the recording's timeline)
      }
      return;
    }
    if (data.frametype !== 1 && data.frametype !== 2) return;      // video
    if (!started) {
      if (data.frametype !== 1) return;                              // start on a key frame
      started = true;
      TS.setCodec(data.enc);   // tag the PMT for the actual codec (H.264 sub-stream vs H.265 main)
      log(`camera_stream: ${data.enc} ${data.width}x${data.height} @${data.fps}fps`);
      if (LIVE) { liveStart(); audioStart(); }
    }
    if (rerequested) { log(`camera_stream: video back ${((Date.now() - last) / 1000).toFixed(1)} s after the stall`); rerequested = false; }
    last = Date.now();
    stats.n++; stats.bytes += data.datalen; stats.ts = data.timestamp;
    if (stats.ts0 == null) { stats.ts0 = data.timestamp; stats.wall0 = Date.now(); }
    feed(outs.video, TS.video(data.frametype === 1, data.frameData), data.frametype === 1);
    liveFrame(data.frametype === 1, data.frameData);
  } else if (message === "onconnect") {
    if (data.code !== 0) quit(`connect failed (${data.code})`, REFUSAL.has(data.code) ? REFUSED_EXIT : 1);
    send("login", { sessionKey: key, user: cfg.user, pwd: cfg.pwd, test: cfg.test || false });
  } else if (message === "onloginresult") {
    if (data.code !== 0) quit(`login failed (${data.code})`);
    send("open_stream", { sessionKey: key, channel: 0, streamid: STREAM });
  } else if (message === "onopenstream") {
    if (data.code !== 0) quit(`open stream failed (${data.code})`, REFUSAL.has(data.code) ? REFUSED_EXIT : 1);
    // the camera stalls its video ~every 4-5 min unless asked for the stream again; re-send
    // open_stream every RENEW_S (seamless, same session) to keep it streaming without a gap.
    if (!renew && RENEW_S > 0) { renew = setInterval(() => { log("camera_stream: renewing the stream"); send("open_stream", { sessionKey: key, channel: 0, streamid: STREAM }); }, RENEW_S * 1000); }
    else if (renew) log(`camera_stream: stream renewed (code ${data.code})`);
  } else if (ctl.on(message, data)) {
    // the control socket's replies (SD-card search and replay, camera settings)
  } else if (message === "ondisconnect") {
    quit(`disconnected (${data.code})`);
  }
};

// ---- the control socket (CTRL_SOCK): everything else goes through this same session, since a
// second session to the camera knocks this one off. One request per connection, a JSON line:
//   {"op":"search","from":E,"to":E}  -> {"rec":[begin,end,type]} lines, then {"done":true}
//   {"op":"replay","from":E,"to":E}  -> {"start":E} (the first frame's time), then MPEG-TS
//                                       (H.265 + AAC) of the card's recording until `to`
//   {"op":"setup","str":"..."}       -> {"result":n,"str":"..."}   (camera settings, remote_setup2)
// E = epoch seconds; the camera's card works in local time (epoch + 5:30, CARD_TZ_S).
const CARD_TZ_S = +(process.env.CARD_TZ_S || 19800);
const ctl = (() => {
  const chn = Array(128).fill(0); chn[0] = 1;
  let search = null, replay = null, setup = null;
  const line = (sock, o) => { try { sock.write(JSON.stringify(o) + "\n"); } catch {} };
  function stopReplay(why) {
    const r = replay; if (!r) return;
    replay = null; clearInterval(r.timer);
    send("replay_stop", { sessionKey: key });
    log(`camera_stream: card replay ${why} (${r.frames} frames)`);
    try { r.sock.end(); } catch {}
  }
  function onRecFrame(d) {
    const r = replay; if (!r) return;
    const isVideo = d.frametype === 1 || d.frametype === 2;
    // card frame times are local ms (epoch + 5:30) as far as seen; told apart from epoch ms by size
    const ep = ts => { const s = ts / 1000; return Math.abs(s - CARD_TZ_S - r.from) < Math.abs(s - r.from) ? s - CARD_TZ_S : s; };
    if (r.first == null) {
      if (d.frametype !== 1) return;                                 // start on a key frame
      r.first = d.ts_ms; line(r.sock, { start: ep(d.ts_ms) });
    }
    r.last = Date.now();
    const t = Math.round((d.ts_ms - r.first) * 90) + 90000;
    if (t < 0) return;
    let buf = null;
    if (isVideo) { r.frames++; buf = r.ts.video(d.frametype === 1, d.frameData, t); }
    else if (d.frametype === 0 && d.enc?.startsWith("AAC") && d.frameData[0] === 0xff) buf = r.ts.audio(Buffer.from(d.frameData), t);
    if (buf) r.sock.write(buf);
    if (ep(d.ts_ms) >= r.to) stopReplay("reached the end");
  }
  function on(message, data) {
    if (message === "onsearchrec") { if (search) line(search, { rec: [data.file_begintime - CARD_TZ_S, data.file_endtime - CARD_TZ_S, data.file_type] }); return true; }
    if (message === "onsearchrecend") {
      if (search) { line(search, { done: true }); search.end(); search = null; }
      send("find_file_stop_2", { sessionKey: key }); return true;
    }
    if (message === "onrecvrecframe") { onRecFrame(data); return true; }
    if (message === "onremotesetup") {
      if (setup) { line(setup, { result: data.result, str: String(data.str || "").replace(/\0+$/, "") }); setup.end(); setup = null; }
      return true;
    }
    return false;
  }
  function request(sock, req) {
    if (!started) return line(sock, { error: "no session yet" }), sock.end();
    if (req.op === "search") {
      if (search) return line(sock, { error: "busy" }), sock.end();
      search = sock;
      send("find_file_start_2", { sessionKey: key, chnlist: chn, begintime: Math.floor(req.from) + CARD_TZ_S, endtime: Math.ceil(req.to) + CARD_TZ_S, type: 15 });
      setTimeout(() => { if (search === sock) { line(sock, { error: "search timed out" }); sock.end(); search = null; } }, 30000);
    } else if (req.op === "replay") {
      if (replay) return line(sock, { error: "busy" }), sock.end();
      if (rerequested) return line(sock, { error: "the live stream is stalled" }), sock.end();
      const r = replay = { sock, from: req.from, to: req.to, ts: makeTS(), first: null, last: Date.now(), frames: 0 };
      sock.on("close", () => { if (replay === r) stopReplay("cancelled"); });
      r.timer = setInterval(() => { if (Date.now() - r.last > 30000) stopReplay("stalled"); }, 1000);
      log(`camera_stream: card replay of ${Math.round(req.to - req.from)} s`);
      send("replay_start", { sessionKey: key, channel: chn, starttime: Math.floor(req.from) + CARD_TZ_S, endtime: Math.ceil(req.to) + CARD_TZ_S, type: 15, tasktype: 1 });
    } else if (req.op === "setup") {
      if (setup) return line(sock, { error: "busy" }), sock.end();
      setup = sock;
      send("remote_setup2", { sessionKey: key, setupstr: String(req.str) });
      setTimeout(() => { if (setup === sock) { line(sock, { error: "setup timed out" }); sock.end(); setup = null; } }, 15000);
    } else { line(sock, { error: "unknown op" }); sock.end(); }
  }
  if (process.env.CTRL_SOCK) {
    const net = require("net"), path_ = process.env.CTRL_SOCK;
    try { fs.unlinkSync(path_); } catch {}
    net.createServer(sock => {
      let buf = "";
      sock.on("error", () => {});
      sock.on("data", d => {
        if (buf === null) return;
        buf += d; const i = buf.indexOf("\n"); if (i < 0) return;
        const text = buf.slice(0, i); buf = null;
        try { request(sock, JSON.parse(text)); } catch (e) { line(sock, { error: e.message }); sock.end(); }
      });
    }).listen(path_, () => fs.chmodSync(path_, 0o600));
  }
  return { on, stopReplay };
})();

process.on("uncaughtException", e => quit(`error: ${e.message}`));
importScripts("433.connector.js");
setTimeout(() => {
  send("create", { context: {}, sessionKey: key });
  send("connectbykey", { sessionKey: key, ngwDomain: cfg.ngw, ngwPort: cfg.port, productId: pid, deviceId: did,
                         deviceSecret: secret, usehttps: cfg.https !== false, cb: false });
}, 500);  // the worker loads its chunks first
const t0 = Date.now();
setInterval(() => {
  if (!started && Date.now() - t0 > CONNECT_S * 1000) quit("no video");
  // a stall: ask for the stream again in this session first (reconnecting takes ~50 s: the
  // cloud refuses a new session for a while), and give up only if that brings nothing
  if (started && !rerequested && Date.now() - last > REREQUEST_S * 1000) {
    rerequested = true;
    ctl.stopReplay("the live stream stalled");  // live first: a card download shares the camera's upload
    log(`camera_stream: no video for ${REREQUEST_S} s, asking for the stream again`);
    send("open_stream", { sessionKey: key, channel: 0, streamid: STREAM });
  }
  if (started && Date.now() - last > QUIET_S * 1000) quit("video stopped");
}, 1000);
