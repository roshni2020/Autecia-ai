// Run: node tests/dashboard.test.cjs. Browser APIs are simulated; no real media or API calls.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync('frontend/dashboard.html','utf8');
class Element {
 constructor(){this.hidden=false;this.disabled=false;this.value='';this.checked=false;this.textContent='';this.attrs={};this.events={};this.children=[];this.classList={toggle(){},add(){},remove(){}};}
 setAttribute(k,v){this.attrs[k]=String(v);} addEventListener(k,v){this.events[k]=v;} focus(){} showModal(){this.open=true;} close(){this.open=false;}
 set innerHTML(value){this.html=value;this.children=[...value.matchAll(/data-choice="(\d+)"/g)].map(m=>{const e=new Element();e.dataset={choice:m[1]};return e;});} get innerHTML(){return this.html||'';}
 click(){if(!this.disabled)return this.onclick?.();} querySelectorAll(){return this.children;}
}
const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);assert.equal(ids.length,new Set(ids).size);
const elements=Object.fromEntries(ids.map(id=>[id,new Element()]));
const calls=[],tracks=[],voices=[],requests=[];let failProcess=false,mediaResolve=null,visionUpdate,visionStops=0;
let resolveSpeech=null;
const profile={camera_enabled:false,pause_tolerance:'medium',support_mode:'other',suggestion_count:3};
const result={interaction_id:'i1',candidate_detail:[{id:'a',text:'I need my book.'},{id:'b',text:'I need my cup.'}],perception:{objects:[{label:'book'}]},trace_summary:['Perception: book detected','Intent: 2 meanings','Learning: memory checked']};
const document={getElementById:id=>{assert.ok(elements[id],id);return elements[id];},body:new Element(),createElement:()=>({width:0,height:0,getContext:()=>({drawImage(){}}),toDataURL:()=> 'data:image/jpeg;base64,test'})};
const context={document,setTimeout,clearTimeout,localStorage:{getItem:()=>null,setItem(){}},console,Number,String,Set,Error,Promise,window:null,navigator:{mediaDevices:{getUserMedia:async()=>{if(mediaResolve)return new Promise(r=>mediaResolve=r);const t={stopped:false,stop(){this.stopped=true;},addEventListener(){}};tracks.push(t);return{getTracks:()=>[t],getVideoTracks:()=>[t]};}}},speechSynthesis:{cancel(){voices.length=0;},speak(u){voices.push(u);}},SpeechSynthesisUtterance:function(text){this.text=text;},Audio:function(){this.play=async()=>{};this.pause=()=>{};},addEventListener(){},DashboardVision:{stop(){visionStops++;},start(v,update){visionUpdate=update;}},fetch:async(path,options)=>{const body=options?.body?JSON.parse(options.body):undefined;calls.push({path,body});let data;
 if(path==='/api/status')data={speech:{enabled:false}};
 else if(path.startsWith('/api/profile'))data={profile};
 else if(path.startsWith('/api/history'))data={memories:[{confirmed_text:'<img src=x>',fragment:'book',success_count:1,failure_count:0}]};
 else if(path==='/interaction/start')data={session_id:'session1'};
 else if(path==='/interaction/process'){if(failProcess) return{ok:false,json:async()=>({detail:'Test failure'})};data=result;}
 else if(path==='/interaction/feedback')data={speak:body.none_fit?null:body.confirmed_text,reflection:{recommendation:'Feedback recorded'}};
 else if(path==='/api/speak'){if(resolveSpeech)return new Promise(r=>resolveSpeech=()=>r({ok:true,json:async()=>({fallback:'browser'})}));data={fallback:'browser'};}
 else throw new Error('Unexpected API '+path);
 return{ok:true,json:async()=>data};}};
context.window=context;vm.createContext(context);vm.runInContext(fs.readFileSync('frontend/dashboard.js','utf8'),context);
const settle=()=>new Promise(r=>setImmediate(r));
(async()=>{
 await settle();assert.equal(elements.connection.textContent,'Connected · Ready to learn');assert.ok(elements.memoryList.innerHTML.includes('&lt;img'));console.log('PASS boot, profile, escaped memories');
 await elements.cameraStart.onclick();assert.equal(elements.cameraEmpty.hidden,true);visionUpdate({objects:['book','cup'],pointing:'cup',hands:true});assert.match(elements.objects.innerHTML,/cup . pointing/);
 elements.transcript.value='I need blue';await elements.suggest.onclick();let request=calls.findLast(c=>c.path==='/interaction/process');assert.deepEqual(request.body.scene_hint,['book','cup']);assert.equal(request.body.pointing_hint,'cup');assert.equal(elements.turns.textContent,1);console.log('PASS camera start, live labels, pointing reaches pipeline, suggestions, insights');
 await elements.yes.onclick();await settle();assert.equal(elements.confirmed.textContent,'I need my book.');assert.equal(elements.yes.disabled,true);assert.equal(elements.rate.textContent,'100%');const count=calls.filter(c=>c.path==='/interaction/feedback').length;await elements.yes.onclick();assert.equal(calls.filter(c=>c.path==='/interaction/feedback').length,count);console.log('PASS confirmation, duplicate-click guard, memory refresh');
 await elements.speak.onclick();assert.equal(voices[0].text,'I need my book.');elements.stop.onclick();assert.equal(voices.length,0);
 resolveSpeech=true;const speaking=elements.speak.onclick();await settle();elements.stop.onclick();resolveSpeech();await speaking;assert.equal(voices.length,0);resolveSpeech=null;console.log('PASS speech fallback and cancellation of pending speech');
 await elements.suggest.onclick();elements.edit.onclick();assert.equal(elements.editDialog.open,true);elements.edited.value='Please bring my water.';elements.editForm.onsubmit({preventDefault(){}});await settle();assert.equal(elements.confirmed.textContent,'Please bring my water.');assert.equal(elements.corrections.textContent,1);console.log('PASS edit dialog and correction');
 await elements.suggest.onclick();await elements.no.onclick();assert.equal(elements.speech.hidden,true);assert.equal(elements.corrections.textContent,2);console.log('PASS rejection does not enable speech');
 failProcess=true;await elements.suggest.onclick();assert.match(elements.notice.textContent,/Test failure/);assert.equal(elements.suggest.disabled,false);failProcess=false;console.log('PASS request failure recovery');
 elements.large.checked=true;elements.large.onchange();elements.contrast.checked=true;elements.contrast.onchange();
 elements.disableDevices.onclick();assert.ok(tracks.every(t=>t.stopped));assert.equal(elements.cameraEmpty.hidden,false);assert.ok(visionStops>0);console.log('PASS accessibility toggles, device shutdown and detector shutdown');
 mediaResolve=true;const starting=elements.cameraStart.onclick();await settle();elements.disableDevices.onclick();const lateTrack={stopped:false,stop(){this.stopped=true;}};mediaResolve({getTracks:()=>[lateTrack],getVideoTracks:()=>[lateTrack]});await starting;assert.equal(lateTrack.stopped,true);mediaResolve=null;console.log('PASS pending camera permission cancelled by devices-off');
 let rec;context.SpeechRecognition=function(){rec=this;this.start=()=>{};this.abort=()=>{this.aborted=true;};};elements.mic.onclick();rec.onresult({results:[[{transcript:'I need the book'}]]});assert.equal(elements.transcript.value,'I need the book');elements.mic.onclick();assert.equal(rec.aborted,true);console.log('PASS microphone transcript and stop');
 elements.end.onclick();assert.equal(elements.turns.textContent,0);assert.equal(elements.speech.hidden,true);assert.match(elements.activityBadge.textContent,/ended/);console.log('PASS end session reset');
 
 context.Blob=Blob;context.FileReader=function(){this.readAsDataURL=()=>{this.result='data:audio/webm;base64,test';this.onload();};};
 const recorders=[];context.MediaRecorder=class {static isTypeSupported(){return true;}constructor(stream,options){this.stream=stream;this.mimeType=options.mimeType;this.state='inactive';recorders.push(this);}start(){this.state='recording';}stop(){this.state='inactive';const done=this.onstop;this.ondataavailable?.({data:new Blob(['audio'])});setImmediate(()=>done?.());}};
 vm.runInContext('serverASR=true',context);
 const processCount=()=>calls.filter(c=>c.path==='/interaction/process').length;
 for(const stop of ['micOff','disableDevices','end']){
   elements.mic.onclick();await settle();assert.equal(recorders.at(-1).state,'recording');assert.equal(elements.suggest.disabled,true);
   const count=processCount();elements[stop].onclick();await settle();assert.equal(recorders.at(-1).state,'inactive');assert.equal(processCount(),count);assert.ok(tracks.every(t=>t.stopped));
 }
 mediaResolve=true;elements.mic.onclick();await settle();elements.end.onclick();const pendingTrack={stopped:false,stop(){this.stopped=true;}};mediaResolve({getTracks:()=>[pendingTrack]});await settle();assert.equal(pendingTrack.stopped,true);mediaResolve=null;
 elements.transcript.value='';elements.mic.onclick();await settle();elements.mic.onclick();await settle();await settle();assert.equal(calls.findLast(c=>c.path==='/interaction/process').body.audio,'data:audio/webm;base64,test');assert.equal(elements.mic.attrs['aria-pressed'],'false');assert.ok(tracks.every(t=>t.stopped));
 elements.end.onclick();
 elements.mic.onclick();await settle();const cancelledCount=processCount();elements.mic.onclick();elements.micOff.onclick();await settle();await settle();assert.equal(processCount(),cancelledCount);
 failProcess=true;elements.transcript.value='';elements.mic.onclick();await settle();elements.mic.onclick();await settle();await settle();assert.equal(elements.suggest.disabled,false);failProcess=false;await elements.suggest.onclick();assert.equal(calls.findLast(c=>c.path==='/interaction/process').body.audio,'data:audio/webm;base64,test');
 elements.end.onclick();
 const Recorder=context.MediaRecorder;context.MediaRecorder=class {static isTypeSupported(){return true;}constructor(){throw new Error('Unsupported recorder');}};elements.mic.onclick();await settle();assert.ok(tracks.every(t=>t.stopped));assert.equal(elements.mic.disabled,false);context.MediaRecorder=Recorder;
 console.log('PASS delayed recording cancellation, audio retry after failure, constructor cleanup');
 console.log('PASS server recording submission, microphone-off, devices-off, session-end, pending permission cancellation');
 console.log('All dashboard control tests passed (simulated browser).');
})().catch(e=>{console.error(e);process.exitCode=1;});

