// Echo's 3D body. TalkingHead (MIT) renders a Ready Player Me human and lip-syncs
// from word timings. Mood is fixed to neutral: Echo's face never performs emotion.
import { TalkingHead } from "talkinghead";

const DEFAULT_AVATAR = "https://cdn.jsdelivr.net/gh/met4citizen/TalkingHead@1.7/avatars/brunette.glb";
let head = null, ready = false, loading = null;

async function init(url) {
  const node = document.getElementById("avatar3d");
  if (!node) return false;
  if (loading) return loading;
  loading = (async () => {
    try {
      head = new TalkingHead(node, { lipsyncModules: ["en"], cameraView: "head", avatarMood: "neutral",
                                     cameraRotateEnable: false, cameraZoomEnable: false, cameraPanEnable: false });
      await head.showAvatar({ url: url || localStorage.getItem("echoloop_avatar") || DEFAULT_AVATAR,
                              body: "F", avatarMood: "neutral", lipsyncLang: "en" });
      node.querySelector(".avatar-loading")?.remove();
      ready = true;
      head.lookAtCamera(1500);
      return true;
    } catch (e) {
      console.error("3D avatar unavailable, using drawn Echo", e);
      node.classList.add("hidden");
      document.querySelector(".echo")?.classList.remove("hidden");
      return false;
    }
  })();
  return loading;
}

async function speak(b64, words, wtimes, wdurations) {
  if (!ready) return false;
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const audio = await head.audioCtx.decodeAudioData(bytes.buffer);
  await head.audioCtx.resume();
  head.speakAudio({ audio, words, wtimes, wdurations });
  return true;
}

// Browser-voice fallback: estimate word timings so the mouth still moves in step.
function speakSilent(text, rate = 1) {
  if (!ready) return false;
  const words = text.split(/\s+/).filter(Boolean);
  const wdurations = words.map((w) => Math.round((90 + 65 * w.length) / rate));
  const wtimes = []; let t = 0;
  for (const d of wdurations) { wtimes.push(t); t += d + 40; }
  const audio = head.audioCtx.createBuffer(1, Math.ceil(head.audioCtx.sampleRate * (t / 1000 + 0.2)), head.audioCtx.sampleRate);
  head.speakAudio({ audio, words, wtimes, wdurations });
  return true;
}

function stop() { if (ready) head.stopSpeaking(); }
function attend(ms = 2000) { if (ready) head.lookAtCamera(ms); }

window.EchoHead = { init, speak, speakSilent, stop, attend, isReady: () => ready };
init();
