/* Dashboard camera analysis. One inference at a time; stopping invalidates pending work. */
window.DashboardVision = (() => {
  const root = 'https://cdn.jsdelivr.net/npm/';
  const urls = {
    tf: root + '@tensorflow/tfjs@4.22.0/dist/tf.min.js',
    coco: root + '@tensorflow-models/coco-ssd@2.2.3/dist/coco-ssd.min.js',
    hands: root + '@mediapipe/hands@0.4.1675469240/hands.js'
  };
  let detector, handModel, loading, timer, generation = 0, ray = null, overlay;
  const scripts = new Map();
  function script(url) {
    if (!scripts.has(url)) scripts.set(url, new Promise((resolve,reject) => {
      const element = document.createElement('script'); element.src = url;
      element.onload = resolve; element.onerror = () => { element.remove(); scripts.delete(url); reject(new Error('Could not download the object detector.')); };
      document.head.appendChild(element);
    }));
    return scripts.get(url);
  }
  async function load() {
    if (loading) return loading;
    loading = (async () => {
      await script(urls.tf); await script(urls.coco); await tf.ready();
      detector = detector || await cocoSsd.load({base:'lite_mobilenet_v2'});
      try {
        await script(urls.hands);
        handModel = new Hands({locateFile:file => root + '@mediapipe/hands@0.4.1675469240/' + file});
        handModel.setOptions({maxNumHands:1,modelComplexity:0,minDetectionConfidence:.6,minTrackingConfidence:.5});
        handModel.onResults(result => {
          const lm = result.multiHandLandmarks?.[0]; ray = null;
          if (!lm) return;
          if (Math.hypot(lm[8].x-lm[0].x,lm[8].y-lm[0].y) <= 1.15*Math.hypot(lm[12].x-lm[0].x,lm[12].y-lm[0].y)) return;
          ray = {x:lm[8].x,y:lm[8].y,dx:lm[8].x-lm[5].x,dy:lm[8].y-lm[5].y};
        });
      } catch { handModel = null; }
    })().catch(error => { loading = null; throw error; });
    return loading;
  }
  function target(objects, width, height) {
    if (!ray) return null;
    const fx=ray.x*width,fy=ray.y*height,dx=ray.dx*width,dy=ray.dy*height,n=Math.hypot(dx,dy)||1;
    let best=null,score=0;
    for (const object of objects) {
      const [x,y,w,h]=object.bbox, vx=x+w/2-fx, vy=y+h/2-fy, distance=Math.hypot(vx,vy)||1;
      const cosine=(vx*dx+vy*dy)/(distance*n);
      const rank=fx>=x&&fx<=x+w&&fy>=y&&fy<=y+h ? 2 : cosine>.75 ? cosine-distance/4000 : 0;
      if(rank>score){score=rank;best=object;}
    }
    return best;
  }
  function stop() { generation++; clearTimeout(timer); ray=null; if(overlay) overlay.getContext('2d').clearRect(0,0,overlay.width,overlay.height); }
  async function start(video, update, error) {
    stop(); const current=generation;
    try { await load(); } catch(e) { if(current===generation) error(e.message); return; }
    if(current!==generation) return;
    if(!overlay){overlay=document.createElement('canvas');overlay.className='vision-overlay';overlay.setAttribute('aria-hidden','true');video.parentElement.appendChild(overlay);}
    let failures=0;
    async function tick(){
      if(current!==generation) return;
      if(!video.videoWidth || document.hidden){timer=setTimeout(tick,400);return;}
      try {
        const objects=(await detector.detect(video,8,.45)).filter(o=>o.class!=='person');
        if(current!==generation) return;
        if(handModel){try{await handModel.send({image:video});}catch{handModel=null;ray=null;}}
        if(current!==generation) return;
        const pointed=target(objects,video.videoWidth,video.videoHeight);
        overlay.width=video.videoWidth;overlay.height=video.videoHeight;
        const g=overlay.getContext('2d');g.lineWidth=3;g.font='600 18px Segoe UI';
        for(const o of objects){const [x,y,w,h]=o.bbox;g.strokeStyle=o===pointed?'#7cddff':'#8eedc1';g.strokeRect(x,y,w,h);const label=o.class+(o===pointed?' · pointing':'');g.fillStyle=g.strokeStyle;g.fillRect(x,Math.max(0,y-27),g.measureText(label).width+14,27);g.fillStyle='#14304b';g.fillText(label,x+7,Math.max(19,y-7));}
        failures=0;
        update({objects:objects.map(o=>o.class),pointing:pointed?.class||null,hands:!!handModel});
      }catch(e){failures++;if(failures===1)error('Live detection paused. Retrying; video is still available.');}
      if(current===generation) timer=setTimeout(tick,failures?2000:500);
    }
    tick();
  }
  return {start,stop};
})();
