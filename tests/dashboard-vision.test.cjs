// Run: node tests/dashboard-vision.test.cjs. Simulated detector and camera frames.
const vm=require('node:vm'),fs=require('node:fs'),assert=require('node:assert/strict');
let resolveLoad, timer, predictions=[{class:'person',bbox:[0,0,50,50]},{class:'book',bbox:[10,10,20,30]}],update=null,errors=[];
let resolveDetection=null;
const canvas={setAttribute(){},getContext:()=>({clearRect(){},strokeRect(){},fillRect(){},fillText(){},measureText:()=>({width:50})})};
const video={videoWidth:640,videoHeight:480,parentElement:{appendChild(){}}};
const ctx={console,Map,Promise,Math,setTimeout:fn=>{timer=fn;return 1;},clearTimeout:()=>{timer=null;},document:{hidden:false,createElement:type=>type==='canvas'?canvas:{remove(){}},head:{appendChild:el=>queueMicrotask(()=>el.onload())}},tf:{ready:async()=>{}},cocoSsd:{load:()=>new Promise(r=>resolveLoad=()=>r({detect:async()=>resolveDetection?new Promise(r=>resolveDetection=r):predictions}))},Hands:function(){this.setOptions=()=>{};this.onResults=()=>{};this.send=async()=>{};}};
ctx.window=ctx;vm.createContext(ctx);vm.runInContext(fs.readFileSync('frontend/dashboard-vision.js','utf8'),ctx);
const settle=()=>new Promise(r=>setImmediate(r));
(async()=>{
 const start=ctx.DashboardVision.start(video,v=>update=v,e=>errors.push(e));await settle();ctx.DashboardVision.stop();resolveLoad();await start;await settle();assert.equal(update,null);console.log('PASS detector loading cancelled by camera stop');
 await ctx.DashboardVision.start(video,v=>update=v,e=>errors.push(e));await settle();assert.equal(update.objects.join(','),'book');assert.ok(timer);console.log('PASS objects update continuously, people excluded');
 update=null;resolveDetection=true;timer();await settle();ctx.DashboardVision.stop();resolveDetection(predictions);await settle();assert.equal(update,null);assert.equal(timer,null);console.log('PASS in-flight detection cannot revive stopped feed');
 console.log('All live-detection lifecycle checks passed.');
})().catch(e=>{console.error(e);process.exitCode=1;});
