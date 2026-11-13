/* Sampled worker poses. Browser kinematics and interpolation never drive physics. */
(function () {
    const assetBase = new URL('.', document.currentScript.src);
    const models = new Map();
    let libraries, current = null, generation = 0;
    const el = (tag, text, cls) => {
        const n = document.createElement(tag);
        if (text !== undefined) n.textContent = text;
        if (cls) n.className = cls;
        return n;
    };
    function unpack(encoded, Type) {
        return new Type(Uint8Array.from(atob(encoded), c => c.charCodeAt(0)).buffer);
    }
    function quaternion(THREE, q) { return new THREE.Quaternion(q[1], q[2], q[3], q[0]); }
    function build(THREE, model) {
        const root = new THREE.Group(), bodies = [], joints = [], objects = [];
        const geometries = new Set(), materials = new Map(), meshes = new Map();
        model.bodies.forEach((body, b) => {
            const group = new THREE.Group();
            group.position.fromArray(body.pos); group.quaternion.copy(quaternion(THREE, body.quat));
            (b ? bodies[body.parent] : root).add(group);
            let attachment = group;
            for (const joint of model.joints.filter(j => j.body === b)) {
                if (joint.kind !== 3) throw new Error('Live preview supports fixed-base hinge-joint robots.');
                const pivot = new THREE.Group(), rotation = new THREE.Group(), back = new THREE.Group();
                pivot.position.fromArray(joint.pos); back.position.fromArray(joint.pos).multiplyScalar(-1);
                attachment.add(pivot); pivot.add(rotation); rotation.add(back); attachment = back;
                joints.push({object:rotation,axis:new THREE.Vector3(...joint.axis),qpos:joint.qpos,zero:model.qpos0[joint.qpos]});
            }
            bodies.push(attachment);
        });
        for (const geom of model.geoms) {
            let geometry;
            const size = geom.size;
            if (geom.kind === 7) {
                if (!meshes.has(geom.mesh)) {
                    const source = model.meshes[geom.mesh];
                    geometry = new THREE.BufferGeometry();
                    geometry.setAttribute('position',new THREE.BufferAttribute(unpack(source.vertices,Float32Array),3));
                    geometry.setIndex(new THREE.BufferAttribute(unpack(source.faces,Uint32Array),1));
                    geometry.computeVertexNormals(); meshes.set(geom.mesh,geometry);
                }
                geometry = meshes.get(geom.mesh);
            } else if (geom.kind === 6) geometry = new THREE.BoxGeometry(2*size[0],2*size[1],2*size[2]);
            else if (geom.kind === 5) {
                geometry = new THREE.CylinderGeometry(size[0],size[0],2*size[1],20); geometry.rotateX(Math.PI/2);
            } else if (geom.kind === 3) {
                geometry = new THREE.CapsuleGeometry(size[0],2*size[1],6,12); geometry.rotateX(Math.PI/2);
            } else {
                geometry = new THREE.SphereGeometry(1,20,12);
                geometry.scale(size[0],geom.kind === 4 ? size[1] : size[0],geom.kind === 4 ? size[2] : size[0]);
            }
            geometries.add(geometry);
            const rgba = geom.rgba, key = rgba.join(',');
            if (!materials.has(key)) materials.set(key,new THREE.MeshStandardMaterial({
                color:new THREE.Color(rgba[0],rgba[1],rgba[2]),roughness:.65,metalness:.1,
                opacity:rgba[3],transparent:rgba[3]<1}));
            const object = new THREE.Mesh(geometry,materials.get(key));
            object.position.fromArray(geom.pos); object.quaternion.copy(quaternion(THREE,geom.quat));
            bodies[geom.body_id].add(object); objects.push(object);
        }
        return {root,objects,apply(q) {
            joints.forEach(j => j.object.quaternion.setFromAxisAngle(j.axis,q[j.qpos]-j.zero));
            root.updateMatrixWorld(true);
        },dispose() {geometries.forEach(g=>g.dispose());materials.forEach(m=>m.dispose());}};
    }
    // Also exercised against MuJoCo kinematics by the numerical regression check.
    window.armLiveKinematics = {build};
    function stop(c) {
        c.active = false; c.epoch = (c.epoch || 0)+1; clearTimeout(c.timer); cancelAnimationFrame(c.frame);
        if (c.abort) c.abort.abort();
    }
    function resume(c) {
        if (!c || !c.ready || c.active || !c.enabled || document.hidden) return;
        c.active = true; c.epoch = (c.epoch || 0)+1; c.poll(); c.frame = requestAnimationFrame(c.animate);
    }
    document.addEventListener('visibilitychange', () => {
        if (!current) return;
        if (document.hidden) stop(current); else resume(current);
    });
    window.armLivePreview = {show: async function(path, enabled) {
        const host = document.getElementById('runner-arm');
        if (!host) return '';
        const visible = Boolean(path && enabled);
        host.style.display = visible ? '' : 'none';
        document.getElementById('runner-visuals').classList.toggle('has-preview',visible);
        if (current && current.path !== path) {
            ++generation; stop(current); current.dispose(); current = null; host.replaceChildren();
        }
        if (current) {
            current.enabled = visible;
            if (!visible) stop(current); else resume(current);
            return 'ready';
        }
        if (!visible) return '';
        const ticket = ++generation;
        const c = current = {path,enabled:true,active:false,ready:false,dispose:()=>{}};
        host.replaceChildren(el('p','Loading live arm preview…','note'));
        // Request samples immediately, even while the one-time mesh download is loading.
        if(!document.hidden)fetch(new URL('../runner-live',assetBase),{method:'POST',headers:{'Content-Type':'application/json'},
            body:JSON.stringify({run:path}),cache:'no-store'}).catch(()=>{});
        try {
            libraries ||= Promise.all([import(new URL('vendor/three.module.js',assetBase)),import(new URL('vendor/OrbitControls.js',assetBase))]);
            const [THREE,{OrbitControls}] = await libraries;
            if (!models.has(path)) {
                const url = new URL('../replay-model',assetBase); url.searchParams.set('run',path);
                models.set(path,fetch(url).then(r=>{if(!r.ok)throw new Error('Saved robot model is unavailable.');return r.json();})
                    .catch(error=>{models.delete(path);throw error;}));
                if (models.size>3) models.delete(models.keys().next().value);
            }
            const model = await models.get(path);
            if (ticket !== generation) return '';
            host.replaceChildren();
            const toolbar=el('div',undefined,'live-arm-toolbar'), reset=el('button','Reset view');
            const status=el('span','Waiting for a worker pose…','note');toolbar.append(status,reset);
            const viewport=el('div',undefined,'live-arm-viewport');
            viewport.setAttribute('aria-label','Live robot preview; drag to rotate, scroll to zoom');
            const note=el('p','One worker · up to 5 pose updates/s · motion between samples is interpolated.','note');
            host.append(toolbar,viewport,note);
            const scene=new THREE.Scene();scene.background=new THREE.Color('#101b29');
            const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));
            viewport.append(renderer.domElement);
            const camera=new THREE.PerspectiveCamera(40,1,.001,100);camera.up.set(0,0,1);
            const controls=new OrbitControls(camera,renderer.domElement);
            scene.add(new THREE.HemisphereLight(0xe8f4ff,0x697b87,2.4));
            const light=new THREE.DirectionalLight(0xffffff,2.5);light.position.set(2,-3,4);scene.add(light);
            const robot=build(THREE,model);scene.add(robot.root);robot.apply(model.qpos0);
            const grid=new THREE.GridHelper(3,20,0x35516b,0x26394a);grid.rotation.x=Math.PI/2;scene.add(grid);
            const axes=new THREE.AxesHelper(.15);scene.add(axes);
            function render() {if(c.enabled&&!document.hidden&&viewport.clientWidth)renderer.render(scene,camera);}
            function fit() {
                const box=new THREE.Box3().setFromObject(robot.root), center=box.getCenter(new THREE.Vector3());
                const span=Math.max(box.getSize(new THREE.Vector3()).length(),.2);
                const distance=span/(2*Math.tan(THREE.MathUtils.degToRad(camera.fov/2)))*Math.max(1,1/camera.aspect);
                camera.position.copy(center).add(new THREE.Vector3(1,-1.4,.8).normalize().multiplyScalar(distance*1.6));
                controls.target.copy(center);controls.update();render();
            }
            const resize=new ResizeObserver(()=>{
                if(!viewport.clientWidth||!viewport.clientHeight)return;
                renderer.setSize(viewport.clientWidth,viewport.clientHeight);
                camera.aspect=viewport.clientWidth/viewport.clientHeight;camera.updateProjectionMatrix();render();
            });resize.observe(viewport);
            controls.addEventListener('change',render);reset.onclick=fit;
            let displayed=model.qpos0.slice(), from=displayed.slice(), target=displayed.slice(), received=0, token=null, stamp=null, fitted=false, lastFrame=0;
            c.animate=function(now) {
                if(!c.active)return;
                if(now-lastFrame>=1000/30) {
                    lastFrame=now;const alpha=Math.min(1,(now-received)/200);
                    if(target.some((q,i)=>Math.abs(q-displayed[i])>1e-10)) {
                        displayed=target.map((q,i)=>from[i]+(q-from[i])*alpha);
                        robot.apply(displayed);render();
                    }
                    if(c.finished && alpha===1)return;
                }
                c.frame=requestAnimationFrame(c.animate);
            };
            c.poll=async function() {
                if(!c.active)return;
                const epoch=c.epoch, controller=new AbortController();c.abort=controller;
                const deadline=setTimeout(()=>controller.abort(),2000);
                try {
                    const response=await fetch(new URL('../runner-live',assetBase),{method:'POST',headers:{'Content-Type':'application/json'},
                        body:JSON.stringify({run:path}),cache:'no-store',signal:controller.signal});
                    if(!response.ok)throw new Error('Live pose unavailable.');
                    const data=await response.json();if(!c.active||epoch!==c.epoch)return;
                    c.finished=data.status!=='running'||data.supported===false;
                    if(data.supported===false) {
                        status.textContent='Live preview is available for newly started tests. This run predates it.';
                        return;
                    }
                    const pose=data.pose;
                    if(pose && pose.q_rad.length===model.qpos0.length && pose.q_rad.every(Number.isFinite)) {
                        if(stamp!==pose.published_at) {
                            const nextToken=pose.worker+':'+pose.segment;
                            target=pose.q_rad;from=token===nextToken?displayed.slice():target.slice();
                            token=nextToken;stamp=pose.published_at;received=performance.now();
                            if(!fitted){robot.apply(target);fit();fitted=true;}
                        }
                        const age=Date.now()/1000-pose.published_at;
                        status.textContent=(data.status==='running'?(age>2?'Waiting · last pose':'Live'):'Last sampled pose · '+data.status)+
                            ' · worker '+pose.worker+' · '+pose.phase+' · t='+pose.time_s.toFixed(3)+' s';
                    } else status.textContent=data.status==='running'?'Waiting for a worker pose…':'No live pose saved for this run. Start a new test with the preview open.';
                } catch(error) {
                    if(c.active&&epoch===c.epoch)status.textContent='Preview reconnecting…';
                } finally {
                    clearTimeout(deadline);if(c.abort===controller)c.abort=null;
                    if(c.active&&epoch===c.epoch&&!c.finished)c.timer=setTimeout(c.poll,200);
                }
            };
            c.dispose=()=>{stop(c);resize.disconnect();controls.dispose();robot.dispose();
                grid.geometry.dispose();grid.material.dispose();axes.geometry.dispose();axes.material.dispose();
                renderer.dispose();renderer.forceContextLoss();};
            c.ready=true;fit();resume(c);
            return 'ready';
        } catch(error) {
            if(ticket===generation){stop(c);c.dispose();current=null;host.replaceChildren(el('p','Live preview unavailable: '+error.message,'note'));}
            return '';
        }
    }};
})();
