// == SPY SCRIPT 16: ROBUST SURVEYOR ==
(function() {
    const payload = function() {
        console.log("%c[Spy] v16 ROBUST SURVEYOR. Ready.", "color: lime; font-weight: bold");

        let trackingObj = null;
        let ballChild = null; // The specific graphic for the ball
        let candidates = new Map(); 
        let history = [];
        let isRecording = false;

        // --- RENDERER HOOK ---
        const hookRender = (Renderer) => {
            if (!Renderer) return;
            const originalRender = Renderer.prototype.render;
            
            Renderer.prototype.render = function(stage, ...args) {
                try {
                    // 1. WATCHDOG: Check if our target is dead
                    if (trackingObj) {
                        // If parent is null, it was removed from the scene (Menu/Death)
                        // If transform is missing, it's destroyed
                        if (!trackingObj.parent || !trackingObj.transform) {
                            console.log("[Spy] Target lost (Destroyed). Rescanning...");
                            resetTracker();
                        }
                    }

                    // 2. SCAN: If no target, look for movement
                    if (!trackingObj) {
                        scanForMovement(stage);
                    }

                    // 3. RECORD: If target exists and recording is active
                    if (isRecording && trackingObj) {
                        // Force update
                        if(trackingObj.updateTransform) trackingObj.updateTransform();
                        
                        let pos = getPos(trackingObj);
                        let now = Date.now();
                        
                        // SMART SIZE: Use the ballChild if found, otherwise fallback
                        let w = 0, h = 0;
                        if (ballChild) {
                            w = ballChild.width || 0;
                            h = ballChild.height || 0;
                        } else {
                            w = trackingObj.width || 0;
                            h = trackingObj.height || 0;
                        }

                        // Save Frame: [Time, X, Y, Width, Height]
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

        function resetTracker() {
            trackingObj = null;
            ballChild = null;
            candidates.clear();
            if (isRecording) {
                console.log("[Spy] Session interrupted by object death. Downloading partial data...");
                isRecording = false;
                downloadCSV();
            }
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

                        // Lock Condition
                        if (prev.score > 20) {
                            lockTarget(node);
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

        function lockTarget(node) {
            trackingObj = node;
            candidates.clear();
            
            // SMART CHILD FINDER
            // Look for the child that is most "Square" (Aspect Ratio ~ 1.0)
            // The text will be wide (Aspect Ratio >> 1.0)
            let bestRatio = 999;
            if (node.children) {
                for (let c of node.children) {
                    if (c.width > 0 && c.height > 0) {
                        let ratio = c.width / c.height;
                        let distFromSquare = Math.abs(1.0 - ratio);
                        
                        // We assume the ball is roughly square dimensions and > 5px
                        if (distFromSquare < bestRatio && c.width > 5) {
                            bestRatio = distFromSquare;
                            ballChild = c;
                        }
                    }
                }
            }

            console.log("%c[Spy] LOCKED ONTO PLAYER!", "color: cyan; font-size: 16px; font-weight:bold");
            if (ballChild) {
                console.log(`[Spy] Identified Ball Graphic: ${ballChild.width.toFixed(1)}x${ballChild.height.toFixed(1)}`);
            } else {
                console.warn("[Spy] Could not isolate ball graphic. Using full container size.");
            }
            console.log("--> PRESS 'R' TO RECORD <--");
        }

        function getPos(obj) {
            if (obj.getGlobalPosition) return obj.getGlobalPosition();
            if (obj.worldTransform) return { x: obj.worldTransform.tx, y: obj.worldTransform.ty };
            return { x: 0, y: 0 };
        }

        // --- KEYBIND ---
        window.addEventListener("keydown", (e) => {
            // SHIFT + R = FORCE RESET
            if (e.key.toLowerCase() === "r" && e.shiftKey) {
                console.log("[Spy] Manual Reset Triggered.");
                resetTracker();
                return;
            }

            // R = TOGGLE RECORDING
            if (e.key.toLowerCase() === "r" && !e.shiftKey) {
                if (!trackingObj) {
                    console.log("[Spy] Not locked yet. Move around.");
                    return;
                }

                if (!isRecording) {
                    history = [];
                    isRecording = true;
                    console.log("%c[Spy] RECORDER STARTED.", "background: red; color: white; padding: 5px; font-size: 14px");
                } else {
                    isRecording = false;
                    console.log("%c[Spy] RECORDING STOPPED.", "color: lime");
                    downloadCSV();
                }
            }
        });

        function downloadCSV() {
            if (history.length === 0) { console.log("No data."); return; }
            
            let csvContent = "Time_ms,X_px,Y_px,BallWidth_px,BallHeight_px\n";
            let t0 = history[0][0];
            
            history.forEach(row => {
                let t = row[0] - t0;
                csvContent += `${t},${row[1].toFixed(3)},${row[2].toFixed(3)},${row[3].toFixed(3)},${row[4].toFixed(3)}\n`;
            });

            let blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            let url = URL.createObjectURL(blob);
            let link = document.createElement("a");
            link.href = url;
            link.download = `bonk_data_${Date.now()}.csv`;
            link.click();
        }
    };

    // --- INJECTOR ---
    let frame = document.getElementById("maingameframe");
    if (frame) {
        let script = frame.contentDocument.createElement("script");
        script.textContent = "(" + payload.toString() + ")();";
        frame.contentDocument.head.appendChild(script);
        console.log("Injector v16 ran successfully.");
    } else {
        console.error("Frame not found.");
    }
})();
