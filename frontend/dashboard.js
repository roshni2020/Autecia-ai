'use strict';
const $ = id => document.getElementById(id);
const user = localStorage.getItem('echoloop_user') || 'user_01';
const userPath = encodeURIComponent(user);
let profile = null, session = null, last = null, chosen = 0, camera = null, recognition = null;
let busy = false, feedbackDone = false, confirmed = null, playback = null, speechVersion = 0, epoch = 0;
let cameraGeneration = 0, liveContext = {objects:[],pointing:null};
let stats = { turns: 0, accepted: 0, feedback: 0, corrected: 0 };
const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'This request could not be completed. Please try again.');
  return result;
}
function notice(text) { $('notice').textContent = text; $('notice').hidden = !text; }
function controls() { for (const id of ['yes','edit','no']) $(id).disabled = busy || !last || feedbackDone; $('suggest').disabled = busy; $('mic').disabled = busy; }
function insights() { $('turns').textContent = stats.turns; $('corrections').textContent = stats.corrected; $('rate').textContent = stats.feedback ? Math.round(stats.accepted / stats.feedback * 100) + '%' : '—'; $('rateNote').textContent = stats.feedback ? `${stats.accepted} of ${stats.feedback} feedback responses` : 'Waiting for feedback'; }
async function loadMemory() {
  const history = await api('/api/history/' + userPath);
  $('memoryList').innerHTML = history.memories.slice(0,4).map(m => `<div class="memory-item"><span aria-hidden="true">${m.success_count ? '✓' : '▤'}</span><div><strong>${esc(m.confirmed_text)}</strong><small>${m.success_count + m.failure_count} confirmation${m.success_count + m.failure_count === 1 ? '' : 's'} · from “${esc(m.fragment)}”</small></div></div>`).join('') || '<p class="muted">Your confirmed words will appear here. Start a conversation to create your first memory.</p>';
}
function stopSpeaking() { speechVersion++; window.speechSynthesis?.cancel(); if (playback) { playback.pause(); playback = null; } }
function stopMic() { if (recognition) { recognition.onend = null; recognition.abort(); recognition = null; } $('mic').textContent = '🎙 Start listening'; $('mic').setAttribute('aria-pressed','false'); $('listening').textContent = 'Take your time. Pauses are welcome.'; }
function stopCamera() { cameraGeneration++; window.DashboardVision.stop(); liveContext = {objects:[],pointing:null}; $('objects').innerHTML = '<span>Camera off</span>'; $('visionStatus').textContent = 'Camera off. Detection starts when you enable it.'; $('retryVision').hidden = true; camera?.getTracks().forEach(t => t.stop()); camera = null; $('video').srcObject = null; $('cameraEmpty').hidden = false; $('cameraBadge').textContent = 'Off'; $('cameraBadge').className = 'pill neutral'; $('cameraToggle').textContent = 'Camera off'; $('cameraToggle').setAttribute('aria-label','Turn camera on'); }
async function toggleCamera() {
  if (camera) { stopCamera(); return; }
  $('cameraStart').disabled = true; $('cameraToggle').disabled = true;
  const turn = epoch, cameraTurn = ++cameraGeneration;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({video:{width:{ideal:960}},audio:false});
    if (turn !== epoch || cameraTurn !== cameraGeneration) { stream.getTracks().forEach(t => t.stop()); return; }
    camera = stream; $('video').srcObject = stream; $('cameraEmpty').hidden = true;
    $('cameraBadge').textContent = '● Live'; $('cameraBadge').className = 'pill'; $('cameraToggle').textContent = 'Turn camera off'; $('cameraToggle').setAttribute('aria-label','Turn camera off');
    stream.getVideoTracks()[0].addEventListener('ended', () => { if (camera === stream) stopCamera(); });
    startDetection();
  } catch { notice('Camera access is unavailable. You can type your message and add context manually.'); }
  finally { $('cameraStart').disabled = false; $('cameraToggle').disabled = false; }
}
function startDetection() {
  if(!camera) return;
  const cameraTurn=cameraGeneration;
  $('retryVision').hidden=true; $('visionStatus').textContent='Loading live object detection...';
  window.DashboardVision.start($('video'), context => {
    if(!camera || cameraTurn!==cameraGeneration) return;
    liveContext=context;
    $('objects').innerHTML=[...new Set(context.objects)].map(label=>'<span>'+esc(label)+(label===context.pointing?' - pointing':'')+'</span>').join('') || '<span>No objects detected yet</span>';
    $('visionStatus').textContent='Live detection - '+context.objects.length+' object(s)'+(context.pointing?' - pointing at '+context.pointing:context.hands?' - no pointing target':' - object detection only');
    $('perception').textContent=$('visionStatus').textContent;
    $('retryVision').hidden=true;
  }, message => { if(camera && cameraTurn===cameraGeneration){$('visionStatus').textContent=message;$('retryVision').hidden=false;} });
}
$('retryVision').onclick=startDetection;
$('cameraStart').onclick = toggleCamera; $('cameraToggle').onclick = toggleCamera;
$('disableDevices').onclick = () => { stopCamera(); stopMic(); notice('Camera and microphone are off.'); };
let serverASR = false, mediaRec = null, micStream = null, audioDataUrl = null;
fetch('/api/status').then(r => r.json()).then(st => { serverASR = !!(st.speech && st.speech.enabled && st.speech.whisper && st.speech.ffmpeg); if (serverASR) $('mic').title = 'Server speech recognition (Whisper ' + st.speech.whisper_model + ')'; }).catch(() => {});
function stopRecording() { if (mediaRec && mediaRec.state === 'recording') mediaRec.stop(); }
async function startRecording() {
  micStream = await navigator.mediaDevices.getUserMedia({audio:true});
  const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : 'audio/webm';
  mediaRec = new MediaRecorder(micStream, {mimeType:mime}); const chunks = [];
  mediaRec.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
  mediaRec.onstop = async () => {
    micStream?.getTracks().forEach(t => t.stop()); micStream = null;
    const blob = new Blob(chunks, {type:'audio/webm'});
    audioDataUrl = await new Promise(ok => { const fr = new FileReader(); fr.onload = () => ok(fr.result); fr.readAsDataURL(blob); });
    $('mic').textContent = '🎙 Start listening'; $('mic').setAttribute('aria-pressed','false');
    $('listening').textContent = 'Transcribing on the server…';
    mediaRec = null;
    $('suggest').click();
  };
  mediaRec.start(250);
  $('mic').textContent = '⏹ Stop & suggest'; $('mic').setAttribute('aria-pressed','true');
  $('listening').textContent = 'Recording… press again when you are done. Pauses are welcome.';
  setTimeout(stopRecording, 45000);
}
$('micOff').onclick = () => { stopRecording(); stopMic(); micStream?.getTracks().forEach(t => t.stop()); micStream = null; audioDataUrl = null; $('listening').textContent = 'Microphone is off.'; notice('Microphone is off. Nothing is being recorded.'); };
$('mic').onclick = () => {
  if (serverASR) {
    if (mediaRec) { stopRecording(); return; }
    stopSpeaking();
    startRecording().catch(() => notice('Microphone is unavailable. You can type your message in the transcript box.'));
    return;
  }
  if (recognition) { stopMic(); return; }
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { notice('Speech recognition is unavailable in this browser. Type your message in the transcript box.'); $('transcript').focus(); return; }
  stopSpeaking();
  const rec = new SR(); recognition = rec; rec.lang = 'en-US'; rec.continuous = true; rec.interimResults = true;
  rec.onresult = e => { $('transcript').value = [...e.results].map(r => r[0].transcript).join(' '); };
  rec.onerror = e => notice('Listening stopped: ' + e.error + '. You can still type your message.');
  rec.onend = () => { if (recognition === rec) { recognition = null; $('mic').textContent = 'Start listening'; $('mic').setAttribute('aria-pressed','false'); $('listening').textContent = 'Ready when you are. Press Suggest.'; } };
  try { rec.start(); $('mic').textContent = 'Stop listening'; $('mic').setAttribute('aria-pressed','true'); $('listening').textContent = 'Listening… tap Stop when you are ready.'; } catch { recognition = null; notice('Could not start listening. Please try again.'); }
};
function frame() { const video = $('video'); if (!camera || !video.videoWidth) return null; const canvas = document.createElement('canvas'); canvas.width = 640; canvas.height = Math.round(640 * video.videoHeight / video.videoWidth); canvas.getContext('2d').drawImage(video,0,0,canvas.width,canvas.height); return canvas.toDataURL('image/jpeg',.7); }
function renderChoices() {
  $('candidates').innerHTML = last.candidate_detail.map((candidate,i) => `<button class="choice" aria-pressed="${i === chosen}" data-choice="${i}">${esc(candidate.text)}</button>`).join('');
  $('candidates').querySelectorAll('button').forEach(button => { button.disabled = feedbackDone; button.onclick = () => { chosen = Number(button.dataset.choice); renderChoices(); $('candidates').querySelectorAll('button')[chosen].focus(); }; });
}
$('suggest').onclick = async () => {
  if (busy) return;
  const transcript = $('transcript').value.trim();
  const audio = audioDataUrl; audioDataUrl = null;
  if (!transcript && !audio) { notice('Speak or type a thought first.'); $('transcript').focus(); return; }
  const turn = epoch; stopMic(); stopSpeaking(); confirmed = null; $('speech').hidden = true; last = null; busy = true; controls(); notice('');
  $('candidates').innerHTML = '<p class="muted" role="status">Finding possible meanings…</p>'; $('activityBadge').textContent = 'Working';
  try {
    if (!profile) profile = (await api('/api/profile/' + userPath)).profile;
    if (!session) session = (await api('/interaction/start',{user_id:user})).session_id;
    const objects = [...new Set([...liveContext.objects, ...$('scene').value.split(',').map(s => s.trim()).filter(Boolean)])];
    const updatedProfile = {...profile,camera_enabled:!!camera || !!objects.length};
    await api('/api/profile/' + userPath, updatedProfile); profile = updatedProfile;
    if (turn !== epoch) return;
    const result = await api('/interaction/process',{user_id:user,session_id:session,transcript,audio,frame:frame(),scene_hint:objects,pointing_hint:liveContext.pointing || $('pointing').value.trim() || null});
    if (turn !== epoch) return;
    if (result.transcript) $('transcript').value = result.transcript;          // server ASR wording, unedited
    $('listening').textContent = result.speech_observations ? `Heard · ${result.speech_observations.vad.pause_count} pause(s) · ${result.speech_observations.fragmented ? 'fragmented' : 'complete'}` : 'Ready when you are.';
    last = result; chosen = 0; feedbackDone = false; stats.turns++; insights(); renderChoices();
    if (!camera) $('objects').innerHTML = result.perception.objects.map(o => `<span>${esc(o.label)}</span>`).join('') || '<span>No visual objects in this interaction</span>';
    for (const [id,prefix] of [['perception','Perception'],['intent','Intent'],['learning','Learning']]) $(id).textContent = result.trace_summary.find(s => s.startsWith(prefix))?.replace(/^[^:]+:\s*/,'') || 'Complete';
    $('reflection').textContent = 'Ready for your feedback'; $('activityBadge').textContent = 'Awaiting feedback';
  } catch(e) { notice(e.message || 'Could not reach EchoLoop.'); $('candidates').innerHTML = '<p class="muted">Your words are still in the transcript. Press Suggest to try again.</p>'; $('activityBadge').textContent = 'Try again'; }
  finally { busy = false; controls(); }
};
$('transcript').addEventListener('keydown',e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); $('suggest').click(); } });
async function feedback(body) {
  if (!last || busy || feedbackDone) return;
  busy = true; controls(); $('candidates').querySelectorAll('button').forEach(b => b.disabled = true);
  const interaction = last.interaction_id, turn = epoch;
  try {
    const result = await api('/interaction/feedback',{interaction_id:interaction,...body});
    if (turn !== epoch) return;
    feedbackDone = true; stats.feedback++; if (body.accepted) stats.accepted++; else stats.corrected++; insights();
    $('reflection').textContent = result.reflection.recommendation; $('activityBadge').textContent = 'Feedback saved';
    confirmed = result.speak ? {text:result.speak,interaction_id:interaction} : null;
    $('speech').hidden = !confirmed; $('confirmed').textContent = confirmed ? confirmed.text : '';
    notice(confirmed ? 'Message confirmed. Speak it when you are ready.' : 'Feedback saved. No message will be spoken.');
    loadMemory().catch(() => notice('Feedback saved. Memory list could not refresh.'));
  } catch(e) { notice(e.message); renderChoices(); }
  finally { busy = false; controls(); }
}
$('yes').onclick = () => { if (last) feedback({accepted:chosen === 0,confirmed_text:last.candidate_detail[chosen].text,chosen_candidate_id:last.candidate_detail[chosen].id}); };
$('no').onclick = () => feedback({accepted:false,none_fit:true});
$('edit').onclick = () => { if (!last) return; $('edited').value = last.candidate_detail[chosen].text; $('editDialog').showModal(); $('edited').focus(); };
$('cancelEdit').onclick = () => $('editDialog').close();
$('editForm').onsubmit = e => { e.preventDefault(); const text = $('edited').value.trim(); if (!text) return; $('editDialog').close(); feedback({accepted:false,confirmed_text:text}); };
$('speak').onclick = async () => {
  if (!confirmed) return;
  stopSpeaking(); const version = speechVersion; const message = {...confirmed};
  $('speak').disabled = true;
  try { const result = await api('/api/speak',message); if (version !== speechVersion) return;
    if (result.audio_base64) { playback = new Audio('data:audio/mpeg;base64,' + result.audio_base64); await playback.play(); }
    else if (window.speechSynthesis) { const utterance = new SpeechSynthesisUtterance(message.text); utterance.rate = Number(localStorage.getItem('echoloop_rate') || 1); speechSynthesis.speak(utterance); }
    else notice('Voice output is unavailable in this browser. Your confirmed text is shown above.');
  } catch(e) { notice(e.message); } finally { $('speak').disabled = false; }
};
$('stop').onclick = stopSpeaking;
$('end').onclick = () => { epoch++; stopCamera(); stopMic(); stopSpeaking(); last = null; confirmed = null; session = null; feedbackDone = true; $('speech').hidden = true; $('candidates').innerHTML = '<p class="muted">Session ended. Type a new thought to start another.</p>'; $('transcript').value = ''; $('activityBadge').textContent = 'Session ended'; stats = {turns:0,accepted:0,feedback:0,corrected:0}; insights(); controls(); notice('Session ended. Camera, microphone, and speech are off.'); };
for (const [id,cls,key] of [['large','large','echoloop_large-text'],['contrast','contrast','echoloop_high-contrast']]) { const input=$(id); input.checked=localStorage.getItem(key)==='1'; document.body.classList.toggle(cls,input.checked); input.onchange=() => { document.body.classList.toggle(cls,input.checked); localStorage.setItem(key,input.checked?'1':'0'); }; }
$('name').textContent = user; $('initial').textContent = user.slice(0,2).toUpperCase();
(async () => { try { profile=(await api('/api/profile/' + userPath)).profile; await loadMemory(); $('connection').textContent='Connected · Ready to learn'; $('connection').classList.add('online'); } catch { $('connection').textContent='Offline'; notice('Could not connect to the backend. Start EchoLoop and refresh this page.'); } })();
window.addEventListener('pagehide',() => { epoch++; stopMic(); stopCamera(); stopSpeaking(); });
