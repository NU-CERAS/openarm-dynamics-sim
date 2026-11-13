/* Local, GPU-rendered playback of recorded poses. No network or physics per frame. */
(function () {
    const assetBase = new URL('.', document.currentScript.src);
    let generation = 0;
    let dispose = () => {};
    let libraries;
    const models = new Map();
    function element(tag, text, className) {
        const node = document.createElement(tag);
        if (text !== undefined) node.textContent = text;
        if (className) node.className = className;
        return node;
    }
    function unpack(encoded, Type) {
        const bytes = Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
        return new Type(bytes.buffer);
    }
    function matrix(THREE, pose) {
        const p = pose;
        return new THREE.Matrix4().set(p[3], p[4], p[5], p[0], p[6], p[7], p[8], p[1],
            p[9], p[10], p[11], p[2], 0, 0, 0, 1);
    }
    window.armTrialReplay = {load: async function (data) {
        const ticket = ++generation;
        dispose();
        dispose = () => {};
        const host = document.getElementById('trial-player');
        if (!host) return '';
        host.replaceChildren();
        if (!data) return '';
        host.append(element('p', 'Loading recorded robot…', 'note'));
        try {
            libraries ||= Promise.all([import(new URL('vendor/three.module.js', assetBase)),
                                       import(new URL('vendor/OrbitControls.js', assetBase))]);
            const [THREE, {OrbitControls}] = await libraries;
            if (!models.has(data.path)) {
                const url = new URL('../replay-model', assetBase);
                url.searchParams.set('run', data.path);
                models.set(data.path, fetch(url).then(r => {
                    if (!r.ok) throw new Error('The saved robot mesh could not be loaded.');
                    return r.json();
                }).catch(error => { models.delete(data.path); throw error; }));
                // Retain a few robot models, not every run visited in a long session.
                if (models.size > 3) models.delete(models.keys().next().value);
            }
            const model = await models.get(data.path);
            if (ticket !== generation) return '';
            host.replaceChildren();
            const panel = element('section', undefined, 'trial-replay');
            const toolbar = element('div', undefined, 'replay-toolbar');
            const play = element('button', 'Pause');
            const restart = element('button', 'Restart');
            const peak = element('button', 'Jump to peak load');
            const reset = element('button', 'Reset view');
            const speedLabel = element('label', 'Speed');
            const speed = element('select');
            speed.setAttribute('aria-label', 'Playback speed');
            for (const value of [0.1, 0.25, 0.5, 1, 2]) {
                const option = element('option', value + '×');
                option.value = value; option.selected = value === 1; speed.append(option);
            }
            speedLabel.append(speed);
            toolbar.append(play, restart, peak, speedLabel, reset);
            const peakNote = element('p', 'Peak target: ' + data.peak_label + ' · ' + data.mates[data.selected_mate], 'note');
            const layout = element('div', undefined, 'replay-layout');
            const viewport = element('div', undefined, 'replay-viewport');
            viewport.setAttribute('aria-label', 'Interactive 3D recorded trial; drag to rotate, scroll to zoom');
            const readout = element('div', undefined, 'replay-readout');
            const clock = element('p', '', 'replay-clock');
            const slider = element('input');
            slider.type = 'range'; slider.min = 0; slider.max = data.time.length - 1; slider.step = 1;
            slider.setAttribute('aria-label', 'Recorded trial timeline');
            const description = (data.full_history ? 'Full history available. ' : 'Reduced history; unsaved motion samples are unavailable. ') +
                'Playback shows ' + data.time.length + ' recorded poses, including this peak; loads match each displayed pose. ' +
                'XYZ loads act on the highlighted link in its local frame. Drag to rotate; scroll to zoom.';
            const note = element('p', description, 'note');
            if (!data.accepted) panel.append(element('p', 'Rejected trial · ' + (data.reason || 'Not included in accepted results'), 'replay-warning'));
            layout.append(viewport, readout);
            panel.append(toolbar, peakNote, layout, clock, slider, note);
            host.append(panel);

            const scene = new THREE.Scene();
            scene.background = new THREE.Color('#101b29');
            const renderer = new THREE.WebGLRenderer({antialias: true});
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            viewport.append(renderer.domElement);
            const camera = new THREE.PerspectiveCamera(40, 1, 0.001, 100);
            camera.up.set(0, 0, 1);
            const controls = new OrbitControls(camera, renderer.domElement);
            scene.add(new THREE.HemisphereLight(0xe8f4ff, 0x697b87, 2.4));
            const light = new THREE.DirectionalLight(0xffffff, 2.5);
            light.position.set(2, -3, 4); scene.add(light);
            const material = new THREE.MeshStandardMaterial({color: 0x9caebe, roughness: 0.65, metalness: 0.15});
            const highlight = new THREE.MeshStandardMaterial({color: 0x25c9a3, roughness: 0.5, metalness: 0.12});
            const geometryCache = new Map();
            const geometrySet = new Set();
            const objects = [];
            for (const geom of model.geoms) {
                let geometry;
                const size = geom.size;
                if (geom.kind === 7) {
                    if (!geometryCache.has(geom.mesh)) {
                        const source = model.meshes[geom.mesh];
                        geometry = new THREE.BufferGeometry();
                        geometry.setAttribute('position', new THREE.BufferAttribute(unpack(source.vertices, Float32Array), 3));
                        geometry.setIndex(new THREE.BufferAttribute(unpack(source.faces, Uint32Array), 1));
                        geometry.computeVertexNormals();
                        geometryCache.set(geom.mesh, geometry);
                    }
                    geometry = geometryCache.get(geom.mesh);
                } else if (geom.kind === 6) geometry = new THREE.BoxGeometry(2*size[0], 2*size[1], 2*size[2]);
                else if (geom.kind === 5) {
                    geometry = new THREE.CylinderGeometry(size[0], size[0], 2*size[1], 24);
                    geometry.rotateX(Math.PI/2);
                } else if (geom.kind === 3) {
                    geometry = new THREE.CapsuleGeometry(size[0], 2*size[1], 8, 16);
                    geometry.rotateX(Math.PI/2);
                } else {
                    geometry = new THREE.SphereGeometry(1, 24, 16);
                    geometry.scale(size[0], geom.kind === 4 ? size[1] : size[0], geom.kind === 4 ? size[2] : size[0]);
                }
                geometry.computeBoundingBox(); geometrySet.add(geometry);
                const object = new THREE.Mesh(geometry, data.members.includes(geom.body) ? highlight : material);
                object.matrixAutoUpdate = false; scene.add(object); objects.push(object);
            }
            const bounds = new THREE.Box3();
            // Fit the entire recorded sweep so moving tips do not leave the view.
            for (const poses of data.poses) for (let i=0; i<objects.length; i++) {
                bounds.union(objects[i].geometry.boundingBox.clone().applyMatrix4(matrix(THREE, poses[i])));
            }
            const center = bounds.getCenter(new THREE.Vector3());
            const span = Math.max(bounds.getSize(new THREE.Vector3()).length(), 0.1);
            const grid = new THREE.GridHelper(span*1.5, 12, 0x35516b, 0x26394a);
            grid.rotation.x = Math.PI/2; grid.position.set(center.x, center.y, bounds.min.z - 0.005); scene.add(grid);
            const axes = new THREE.AxesHelper(span*0.12); scene.add(axes);
            function render() { renderer.render(scene, camera); }
            function fit() {
                const distance = span / (2*Math.tan(THREE.MathUtils.degToRad(camera.fov/2))) * Math.max(1, 1/camera.aspect);
                camera.position.copy(center).add(new THREE.Vector3(1, -1.4, 0.8).normalize().multiplyScalar(distance*1.15));
                camera.near = span/1000; camera.far = span*100; camera.updateProjectionMatrix();
                controls.target.copy(center); controls.update(); render();
            }
            const resize = new ResizeObserver(() => {
                const width = viewport.clientWidth, height = viewport.clientHeight;
                if (!width || !height) return;
                renderer.setSize(width, height); camera.aspect = width/height; camera.updateProjectionMatrix(); render();
            });
            resize.observe(viewport);
            renderer.setSize(viewport.clientWidth, viewport.clientHeight);
            camera.aspect = viewport.clientWidth/viewport.clientHeight;
            controls.addEventListener('change', render);
            reset.onclick = fit;
            const cells = [];
            for (const mate of data.mates) {
                readout.append(element('h3', mate));
                const table = element('table');
                const head = element('tr');
                for (const title of ['Component', 'Recorded value']) head.append(element('th', title));
                table.append(head);
                const values = [];
                for (const label of ['Fx (N)', 'Fy (N)', 'Fz (N)', 'Mx (Nm)', 'My (Nm)', 'Mz (Nm)', '|F| (N)', '|M| (Nm)']) {
                    const row = element('tr'), value = element('td');
                    row.append(element('th', label), value); table.append(row); values.push(value);
                }
                cells.push(values); readout.append(table);
            }
            let frame = -1, playing = false, requestId = null, lastTime = 0, playhead = data.time[0];
            function show(index) {
                if (index === frame) return;
                frame = index;
                objects.forEach((object, i) => { object.matrix.copy(matrix(THREE, data.poses[index][i])); object.matrixWorldNeedsUpdate = true; });
                for (let mate=0; mate<cells.length; mate++) {
                    const load = data.loads[index][mate];
                    const values = [...load, Math.hypot(...load.slice(0,3)), Math.hypot(...load.slice(3,6))];
                    values.forEach((v,j) => { cells[mate][j].textContent = (j<6 && v>=0 ? '+' : '') + v.toPrecision(5); });
                }
                slider.value = index;
                slider.setAttribute('aria-valuetext', data.time[index].toFixed(4) + ' seconds');
                clock.textContent = 'Recorded time ' + data.time[index].toFixed(4) + ' / ' + data.time.at(-1).toFixed(4) + ' s' +
                    (index === data.peak_index ? ' · selected peak' : '');
                render();
            }
            function pause() {
                playing = false; play.textContent = 'Play'; cancelAnimationFrame(requestId);
            }
            function animate(now) {
                if (!playing) return;
                playhead += (now-lastTime)/1000 * Number(speed.value); lastTime = now;
                // Hold the latest recorded sample; do not fabricate intermediate loads.
                let lo=0, hi=data.time.length-1;
                while (lo<hi) { const mid=Math.ceil((lo+hi)/2); if (data.time[mid]<=playhead) lo=mid; else hi=mid-1; }
                show(lo);
                if (playhead >= data.time.at(-1)) { show(data.time.length-1); pause(); }
                else requestId = requestAnimationFrame(animate);
            }
            function start() {
                if (playing) return;
                if (frame === data.time.length-1) { show(0); playhead=data.time[0]; }
                playing = true; play.textContent = 'Pause'; lastTime = performance.now();
                requestId = requestAnimationFrame(animate);
            }
            play.onclick = () => playing ? pause() : start();
            restart.onclick = () => { pause(); show(0); playhead=data.time[0]; start(); };
            peak.onclick = () => { pause(); show(data.peak_index); playhead=data.time[data.peak_index]; };
            slider.oninput = () => { pause(); const index=Number(slider.value); show(index); playhead=data.time[index]; };
            const visibility = () => { if (document.hidden) pause(); };
            document.addEventListener('visibilitychange', visibility);
            dispose = () => {
                pause(); resize.disconnect(); controls.dispose();
                document.removeEventListener('visibilitychange', visibility);
                geometrySet.forEach(g => g.dispose()); material.dispose(); highlight.dispose();
                grid.geometry.dispose(); grid.material.dispose(); axes.geometry.dispose(); axes.material.dispose();
                renderer.dispose(); renderer.forceContextLoss();
            };
            show(0); fit();
            panel.scrollIntoView({block: 'start', behavior: 'instant'});
            start();
            return 'ready';
        } catch (error) {
            if (ticket === generation) {
                dispose(); dispose = () => {};
                host.replaceChildren(element('p', 'Playback unavailable: ' + error.message, 'note'));
            }
            return '';
        }
    }};
})();
