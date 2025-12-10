// == SPY SCRIPT 15: THE DATA SCIENTIST (CSV + ZOOM) ==
(function() {
    const payload = function() {
        console.log("%c[Spy] v15 DATA SCIENTIST. Wait for lock, then Press 'R'.", "color: lime; font-weight: bold");

        let trackingObj = null;
        let candidates = new Map(); 
        let history = [];
        let isRecording = false;

        // --- RENDERER HOOK ---
        const hookRender = (Renderer) => {
            if (!Renderer) return;
            const originalRender = Renderer.prototype.render;
            
            Renderer.prototype.render = function(stage, ...args) {
                try {
                    // 1. Scan for movement
                    if (!trackingObj) {
                        scanForMovement(stage);
                    }

                    // 2. Flight Recorder
                    if (isRecording && trackingObj) {
                        if(trackingObj.updateTransform) trackingObj.updateTransform();
                        
                        let pos = getPos(trackingObj);
                        let now = Date.now();
                        
                        // CAPTURE SIZE (Crucial for normalizing Zoom/PPM)
                        // We try .width/.height (fast) first, fallback to bounds (slow)
                        let w = trackingObj.width || 0;
                        let h = trackingObj.height || 0;
                        
                        // Format: [Time, X, Y, Width, Height]
                        history.push([now, pos.x, pos.y, w, h]);
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
                            console.log("--> PRESS 'R' TO START/STOP RECORDING <--");
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
            if (e.key.toLowerCase() === "r") {
                if (!trackingObj) {
                    console.log("[Spy] Not locked yet. Move around.");
                    return;
                }

                if (!isRecording) {
                    // START
                    history = [];
                    isRecording = true;
                    console.log("%c[Spy] RECORDER STARTED. Perform tests now...", "background: red; color: white; padding: 5px; font-size: 14px");
                } else {
                    // STOP
                    isRecording = false;
                    console.log("%c[Spy] RECORDING STOPPED. Downloading CSV...", "color: lime");
                    downloadCSV();
                }
            }
        });

        function downloadCSV() {
            if (history.length === 0) { console.log("No data."); return; }
            
            // 1. Header
            let csvContent = "Time_ms,X_px,Y_px,Width_px,Height_px\n";
            
            // 2. Data Rows (Normalize start time to 0)
            let t0 = history[0][0];
            history.forEach(row => {
                let t = row[0] - t0;
                let x = row[1].toFixed(3);
                let y = row[2].toFixed(3);
                let w = row[3].toFixed(3);
                let h = row[4].toFixed(3);
                csvContent += `${t},${x},${y},${w},${h}\n`;
            });

            // 3. Create Blob and Download
            let blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            let url = URL.createObjectURL(blob);
            
            let link = document.createElement("a");
            link.setAttribute("href", url);
            link.setAttribute("download", "bonk_physics_data.csv");
            link.style.display = "none";
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    };

    // --- INJECTOR ---
    let frame = document.getElementById("maingameframe");
    if (frame) {
        let script = frame.contentDocument.createElement("script");
        script.textContent = "(" + payload.toString() + ")();";
        frame.contentDocument.head.appendChild(script);
        console.log("Injector v15 ran successfully.");
    } else {
        console.error("Frame not found.");
    }
})();
