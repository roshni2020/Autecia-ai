// Ambient geometry only: the scene never uses camera or microphone input.
import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const host = document.createElement('div');
host.className = 'spatial-scene';
host.setAttribute('aria-hidden', 'true');
document.getElementById('cameraEmpty').prepend(host);
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, powerPreference: 'low-power' });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
  renderer.setClearColor(0x000000, 0);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.25;
  host.append(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(34, 1, .1, 50);
  camera.position.set(0, 0, 7);
  const pmrem = new THREE.PMREMGenerator(renderer);
  const room = new RoomEnvironment();
  const environment = pmrem.fromScene(room, .04);
  scene.environment = environment.texture;
  room.dispose(); pmrem.dispose();
  scene.add(new THREE.HemisphereLight(0xe7f4ff, 0x5352a7, 2));
  const light = new THREE.DirectionalLight(0xffffff, 4);
  light.position.set(-3, 5, 4); scene.add(light);
  const blue = new THREE.PointLight(0x7eabff, 25); blue.position.set(2, -2, 2); scene.add(blue);
  const sculpture = new THREE.Group(); scene.add(sculpture);
  const pearl = new THREE.MeshPhysicalMaterial({ color: 0xb9caff, metalness: .18, roughness: .17, clearcoat: 1, clearcoatRoughness: .12, iridescence: .65, iridescenceIOR: 1.35 });
  const glass = new THREE.MeshPhysicalMaterial({ color: 0x9abaff, metalness: .1, roughness: .13, transmission: .45, thickness: .8, ior: 1.45, clearcoat: 1 });
  const loop = new THREE.Mesh(new THREE.TorusKnotGeometry(.86, .23, 180, 28, 2, 3), pearl);
  loop.rotation.set(.25, -.45, .15); sculpture.add(loop);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(1.53, .035, 12, 120), glass);
  ring.rotation.set(.8, .4, -.3); sculpture.add(ring);
  const orbs = [];
  for (let i = 0; i < 4; i++) {
    const orb = new THREE.Mesh(new THREE.SphereGeometry(.08 + i * .025, 24, 16), i % 2 ? pearl : glass);
    sculpture.add(orb); orbs.push(orb);
  }
  const reduce = matchMedia('(prefers-reduced-motion: reduce)');
  const pointer = { x: 0, y: 0 };
  const panel = document.querySelector('.camera-panel');
  panel.addEventListener('pointermove', e => { const r = panel.getBoundingClientRect(); pointer.x = (e.clientX - r.left) / r.width - .5; pointer.y = (e.clientY - r.top) / r.height - .5; });
  panel.addEventListener('pointerleave', () => { pointer.x = pointer.y = 0; });
  const resize = new ResizeObserver(() => { const w = host.clientWidth, h = host.clientHeight; if (!w || !h) return; renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); });
  resize.observe(host);
  let visible = true, frame = 0, last = 0;
  const observer = new IntersectionObserver(entries => { visible = entries[0].isIntersecting; }); observer.observe(host);
  function draw(now) {
    frame = requestAnimationFrame(draw);
    if (!visible || document.hidden || now - last < 33) return;
    last = now;
    const still = reduce.matches || document.documentElement.classList.contains('still-ui');
    const t = still ? 0 : now * .00025;
    sculpture.rotation.y = Math.sin(t) * .18 + (still ? 0 : pointer.x * .25);
    sculpture.rotation.x = still ? .1 : pointer.y * .15;
    loop.rotation.z = t * .25;
    sculpture.position.y = Math.sin(t * 2) * .08;
    orbs.forEach((orb, i) => { const a = t + i * Math.PI / 2; orb.position.set(Math.cos(a) * 1.65, Math.sin(a) * 1.2, Math.sin(a + .7) * .5); });
    renderer.render(scene, camera);
  }
  frame = requestAnimationFrame(draw);
  window.addEventListener('pagehide', () => { cancelAnimationFrame(frame); resize.disconnect(); observer.disconnect(); scene.traverse(o => o.geometry?.dispose()); pearl.dispose(); glass.dispose(); environment.dispose(); renderer.dispose(); }, { once: true });
} catch (error) {
  renderer?.dispose(); host.remove();
  console.warn('Ambient 3D unavailable; keeping the static interface.', error);
}
