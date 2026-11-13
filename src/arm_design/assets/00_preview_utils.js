/* Dash loads assets alphabetically; shared helpers precede both preview scripts. */
(function () {
    function element(tag, text, className) {
        const node = document.createElement(tag);
        if (text !== undefined) node.textContent = text;
        if (className) node.className = className;
        return node;
    }
    function unpack(encoded, Type) {
        return new Type(Uint8Array.from(atob(encoded), c => c.charCodeAt(0)).buffer);
    }
    function geometry(THREE, model, geom, meshes, radialSegments, capSegments, sphereSegments) {
        const size = geom.size;
        let result;
        if (geom.kind === 7) {
            if (!meshes.has(geom.mesh)) {
                const source = model.meshes[geom.mesh];
                result = new THREE.BufferGeometry();
                result.setAttribute('position', new THREE.BufferAttribute(unpack(source.vertices, Float32Array), 3));
                result.setIndex(new THREE.BufferAttribute(unpack(source.faces, Uint32Array), 1));
                result.computeVertexNormals();
                meshes.set(geom.mesh, result);
            }
            return meshes.get(geom.mesh);
        }
        if (geom.kind === 6) return new THREE.BoxGeometry(2*size[0], 2*size[1], 2*size[2]);
        if (geom.kind === 5) {
            result = new THREE.CylinderGeometry(size[0], size[0], 2*size[1], radialSegments);
            result.rotateX(Math.PI/2);
        } else if (geom.kind === 3) {
            result = new THREE.CapsuleGeometry(size[0], 2*size[1], capSegments, sphereSegments);
            result.rotateX(Math.PI/2);
        } else {
            result = new THREE.SphereGeometry(1, radialSegments, sphereSegments);
            result.scale(size[0], geom.kind === 4 ? size[1] : size[0], geom.kind === 4 ? size[2] : size[0]);
        }
        return result;
    }
    window.armPreviewUtils = {element, geometry};
})();
