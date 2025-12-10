// == SPY SCRIPT 13: SYNTAX FIX ==
(function() {
    // We define the payload as a real function first.
    // This allows us to use backticks inside it without breaking the injection string.
    const payload = function() {
        console.log("%c[Spy] v13 MOTION DETECTOR Active.", "color: lime; font-weight: bold");

        let trackingObj = null;
        let candidates = new Map(); 
        let history = [];
        let recording = false;

        // --- RENDERER HOOK ---
        const hookRender = (Renderer) => {
            if (!Renderer) return;
            const originalRender = Renderer.prototype.render;
            
            Renderer.prototype.render = function(stage, ...args) {
                try {
                    // 1. Scan for movement (if not locked)
                    if (!trackingObj) {
                        scanForMovement(stage);
                    }

                    // 2. Record Data (if locked and recording)
                    if (recording && trackingObj) {
                        if(trackingObj.updateTransform) trackingObj.updateTransform();
                        
                        let pos = getPos(trackingObj);
                        let now = Date.now();
                        
                        // Dedup frames
                        if (history.length === 0 || history[history.length-1].t !== now) {
                            history.push({ t: now, x: pos.x, y: pos.y });
                        }
                    }
                } catch(e) {}

                return originalRender.call(this, stage, ...args);
            };
            console.log("[Spy] Hooked " + Renderer.name);
        };

        if (window.PIXI) {
            if (window.PIXI.Renderer) hookRender(window.PIXI.Renderer);
            if (window.PIXI.WebGLRenderer) hookRender(window.PIXI.WebGLRenderer);
        }

        // --- SCANNER ---
        function scanForMovement(node, depth=0) {
            if (!node || depth > 8) return;

            if (node.children && node.children.length > 0) {
                let pos = getPos(node);

                if (candidates.has(node)) {
                    let prev = candidates.get(node);
                    let dist = Math.abs(pos.x - prev.x) + Math.abs(pos.y - prev.y);

                    if (dist > 0.5 && dist < 100) {
                        prev.score++;
                        prev.x = pos.x;
                        prev.y = pos.y;

                        if (prev.score > 20) {
                            trackingObj = node;
                            console.log("%c[Spy] LOCKED ONTO PLAYER!", "color: cyan; font-size: 16px; font-weight:bold");
                            console.log("--> PRESS 'T' TO MEASURE PHYSICS <--");
                            candidates.clear();
                            return;
                        }
                    }
                } else {
                    candidates.set(node, { x: pos.x, y: pos.y, score: 0 });
                }

                for (let i = 0; i < node.children.length; i++) {
                    scanForMovement(node.children[i], depth + 1);
                    if (trackingObj) return;
                }
            }
        }

        function getPos(obj) {
            if (obj.getGlobalPosition) return obj.getGlobalPosition();
            if (obj.worldTransform) return { x: obj.worldTransform.tx, y: obj.worldTransform.ty };
            return { x: 0, y: 0 };
        }

        // --- KEYBIND ---
        window.addEventListener("keydown", (e) => {
            if (e.key.toLowerCase() === "t") {
                if (!trackingObj) {
                    console.log("[Spy] Still scanning... Move around!");
                    return;
                }
                history = [];
                recording = true;
                console.log("%c[Spy] RECORDING (2s)...", "background: red; color: white");
                
                setTimeout(() => {
                    recording = false;
                    analyze(history);
                }, 2000);
            }
        });

        function analyze(data) {
            if (data.length < 5) { console.log("[Spy] No data."); return; }
            
            let t0 = data[0].t;
            let clean = data.map(d => ({ t: (d.t - t0)/1000, x: d.x, y: d.y }));
            
            let start = clean[0];
            let end = clean[clean.length-1];
            let totalTime = end.t - start.t;

            // Acceleration (End Velocity - Start Velocity)
            let vs_y = (clean[5].y - clean[0].y) / (clean[5].t - clean[0].t);
            let vs_x = (clean[5].x - clean[0].x) / (clean[5].t - clean[0].t);
            
            let ve_y = (clean[clean.length-1].y - clean[clean.length-6].y) / (clean[clean.length-1].t - clean[clean.length-6].t);
            let ve_x = (clean[clean.length-1].x - clean[clean.length-6].x) / (clean[clean.length-1].t - clean[clean.length-6].t);

            let ay = (ve_y - vs_y) / totalTime;
            let ax = (ve_x - vs_x) / totalTime;

            console.log("--- PHYSICS REPORT (v13) ---");
            console.log("Frames Captured: " + clean.length);
            console.log("Duration: " + totalTime.toFixed(2) + "s");
            console.log(`Displacement Y: ${(end.y - start.y).toFixed(2)} px`);
            console.log(`Displacement X: ${(end.x - start.x).toFixed(2)} px`);
            console.log(`RAW GRAVITY (Accel Y): ${ay.toFixed(2)} px/s^2`);
            console.log(`RAW FORCE (Accel X): ${ax.toFixed(2)} px/s^2`);
            console.log("----------------------------");
            // Log a sample so we can debug if numbers are crazy
            console.log("SAMPLE DATA: " + JSON.stringify(clean.slice(0, 10)));
        }
    };

    // --- INJECTOR ---
    let frame = document.getElementById("maingameframe");
    if (frame) {
        let script = frame.contentDocument.createElement("script");
        // We convert the function to a string dynamically. No backtick conflicts!
        script.textContent = "(" + payload.toString() + ")();";
        frame.contentDocument.head.appendChild(script);
        console.log("Injector v13 ran successfully.");
    } else {
        console.error("Frame not found.");
    }
})();
