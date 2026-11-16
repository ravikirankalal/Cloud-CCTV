// Test: the camera's SD card through the cloud (what the TrueCloud portal's Download does).
//   node sdtest.js DEVICE.json            search today's recordings, then download REPLAY_S
// seconds from 30 minutes ago in download mode; reports coverage, speed and the video.
const fs = require("fs"), path = require("path"), vm = require("vm");
const WebSocket = require("ws");
const { XMLHttpRequest } = require("xmlhttprequest-ssl");
const JS = path.join("/home/ubuntu/rtmp-web", "truecloud");
const cfg = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const [pid, did, secret] = cfg.dev.split(":");
const log = (...a) => process.stderr.write(a.join(" ") + "\n");
console.log = console.error = console.warn = console.info = () => {};  // the connector logs keys
const self = globalThis; self.self = self;
self.location = "https://www.cms.truecloudplus.co.in/static/433.connector.js";
self.WebSocket = WebSocket; self.XMLHttpRequest = XMLHttpRequest;
self.importScripts = (...u) => u.forEach(x => vm.runInThisContext(fs.readFileSync(path.join(JS, path.basename(x)), "utf8")));
const send = (message, data) => self.onmessage({ data: { message, data } });
const key = "t", TZ = 330 * 60;                       // local seconds = epoch + 5:30
const local = s => Math.floor(s) + TZ, hm = s => new Date((s - TZ) * 1000).toLocaleTimeString("en-GB", { timeZone: "Asia/Kolkata" });
const chn = Array(128).fill(0); chn[0] = 1;
const REPLAY_S = 60;
let files = [], t0, frames = 0, bytes = 0, first = null, last = null, info = null, out = null;
self.postMessage = ({ message, data }) => {
  if (message === "onconnect") { log("connect", data.code); if (data.code === 0) send("login", { sessionKey: key, user: cfg.user, pwd: cfg.pwd, test: false }); }
  else if (message === "onloginresult") {
    log("login", data.code);
    const now = Date.now() / 1000, day0 = Math.floor((now + TZ) / 86400) * 86400;  // local midnight
    send("find_file_start_2", { sessionKey: key, chnlist: chn, begintime: day0, endtime: local(now), type: 15 });
  } else if (message === "onsearchrec") files.push([data.file_begintime, data.file_endtime, data.file_type]);
  else if (message === "onsearchrecend") {
    files.sort((a, b) => a[0] - b[0]);
    let cover = 0, gaps = [];
    files.forEach((f, i) => { cover += f[1] - f[0]; if (i && f[0] - files[i - 1][1] > 5) gaps.push(`${hm(files[i - 1][1])}-${hm(f[0])}`); });
    const types = {}; files.forEach(f => types[f[2]] = (types[f[2]] || 0) + 1);
    log(`SD card today: ${files.length} files, ${(cover / 3600).toFixed(1)} h recorded, from ${files.length ? hm(files[0][0]) : "-"} to ${files.length ? hm(files.at(-1)[1]) : "-"}; types ${JSON.stringify(types)}; gaps > 5 s: ${gaps.length} ${gaps.slice(0, 8).join(", ")}`);
    const start = local(Date.now() / 1000 - 1800);
    log(`downloading ${REPLAY_S} s from ${hm(start)} (download mode)`);
    t0 = Date.now();
    send("replay_start", { sessionKey: key, channel: chn, starttime: start, endtime: start + REPLAY_S, type: 15, tasktype: 1 });
  } else if (message === "onrecvrecframe") {
    if (data.frametype !== 1 && data.frametype !== 2) return;
    if (!out) { if (data.frametype !== 1) return; info = `${data.enc} ${data.width}x${data.height}`; out = fs.createWriteStream("/tmp/sdtest.h265"); }
    out.write(Buffer.from(data.frameData)); frames++; bytes += data.datalen;
    first ??= data.ts_ms; last = data.ts_ms;
  }
};
importScripts("433.connector.js");
setTimeout(() => { send("create", { context: {}, sessionKey: key });
  send("connectbykey", { sessionKey: key, ngwDomain: cfg.ngw, ngwPort: cfg.port, productId: pid, deviceId: did, deviceSecret: secret, usehttps: cfg.https !== false, cb: false }); }, 500);
setTimeout(() => {
  const wall = t0 ? (Date.now() - t0) / 1000 : 0, span = first != null ? (last - first) / 1000 : 0;
  log(`download: ${frames} frames (${info}), ${(bytes / 1e6).toFixed(1)} MB, covering ${span.toFixed(1)} s of recording, received in ${wall.toFixed(0)} s` +
      (span ? ` (${(span / wall).toFixed(1)}x real time)` : ""));
  process.exit(0);
}, 60000);
