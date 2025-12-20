(function (global) {
  const PX_PER_M = 42;
  const BALL_DIAMETER = 1;
  const BALL_RADIUS = BALL_DIAMETER / 2;
  const GRAVITY = -9.5;
  const UP_THRUST = 5.7;
  const DOWN_THRUST = 5.7;
  const H_THRUST = 5.45;
  const TAP_APEX = 1.2;
  const JUMP_SPEED = Math.sqrt(2 * Math.abs(GRAVITY) * TAP_APEX);
  const PLAYER_RESTITUTION = 0.4;

  const MAPS = [
    {
      id: "mini-course",
      name: "Mini Course",
      world: {
        floorLevel: 0,
        ceiling: 9.2,
        left: -11.5,
        right: 11.5,
      },
      obstacles: [
        { x: -12.5, y: -1, w: 25, h: 1 },
        { x: -12.5, y: 0, w: 1, h: 8 },
        { x: 11.5, y: 0, w: 1, h: 8 },
        { x: -4.5, y: 2.2, w: 5, h: 0.4 },
        { x: 4, y: 3.8, w: 3.5, h: 0.4 },
        { x: -9.5, y: 1.2, w: 2.5, h: 1.4 },
      ],
      spawnPoints: [
        { x: -8, y: 1.3 },
        { x: -6, y: 1.3 },
      ],
      killPlane: -3.5,
    },
    {
      id: "flat-1v1",
      name: "Flat 1v1",
      world: {
        floorLevel: -2,
        ceiling: 9.2,
        left: -11.5,
        right: 11.5,
      },
      obstacles: [{ x: -5, y: 2.8, w: 10, h: 0.35 }],
      spawnPoints: [
        { x: -3.8, y: 3.2 },
        { x: 3.8, y: 3.2 },
      ],
      killPlane: -4.5,
    },
    {
      id: "flat-wide",
      name: "Flat Wide",
      world: {
        floorLevel: -2,
        ceiling: 9.2,
        left: -11.5,
        right: 11.5,
      },
      obstacles: [{ x: -11.5, y: 2.8, w: 23, h: 0.35 }],
      spawnPoints: [
        { x: -8.5, y: 3.2 },
        { x: 8.5, y: 3.2 },
      ],
      killPlane: -5,
    },
    {
      id: "pyramid",
      name: "Pyramid Steps",
      world: {
        floorLevel: -2,
        ceiling: 9.2,
        left: -12,
        right: 12,
      },
      obstacles: [
        { x: -10, y: 1, w: 20, h: 0.5 },
        { x: -7.5, y: 2, w: 15, h: 0.5 },
        { x: -5, y: 3, w: 10, h: 0.5 },
        { x: -2.5, y: 4, w: 5, h: 0.5 },
      ],
      spawnPoints: [
        { x: -9, y: 1.5 },
        { x: 9, y: 1.5 },
      ],
      killPlane: -5,
    },
  ];

  const AI_MODES = [
    { id: "none", name: "None" },
    { id: "simple-xy", name: "Simple XY comparison" },
  ];

  const PLAYER_CONFIGS = [
    {
      label: "P1",
      color: "#d1d5db",
      controls: {
        left: "ArrowLeft",
        right: "ArrowRight",
        up: "ArrowUp",
        down: "ArrowDown",
      },
      defaultAiMode: "none",
    },
    {
      label: "P2",
      color: "#4b5563",
      controls: {
        left: "KeyA",
        right: "KeyD",
        up: "KeyW",
        down: "KeyS",
      },
      defaultAiMode: "none",
    },
  ];

  function defaultOverlayFormatter(players) {
    return players
      .map(
        (player) =>
          `${player.label}: (${player.x.toFixed(2)} m, ${player.y.toFixed(2)} m)` +
          `  v=(${player.vx.toFixed(2)} m/s, ${player.vy.toFixed(2)} m/s)`
      )
      .join("    ");
  }

  function createBlankIntent() {
    return { left: false, right: false, up: false, down: false };
  }

  function createPlayers(configs) {
    return configs.map((config, index) => ({
      label: config.label,
      color: config.color,
      controls: config.controls,
      index,
      x: 0,
      y: 0,
      vx: 0,
      vy: 0,
      touchingGround: false,
      aiMode: config.defaultAiMode ?? "none",
      aiIntent: createBlankIntent(),
    }));
  }

  function createSandbox(options = {}) {
    const canvas = options.canvas;
    const ctx = canvas.getContext("2d");
    const overlay = options.overlay ?? null;
    const mapSelect = options.mapSelect ?? null;
    const aiSelect = options.aiSelect ?? null;
    const velocityToggle = options.velocityToggle ?? null;
    const resetButton = options.resetButton ?? null;

    const maps = options.maps ?? MAPS;
    const playerConfigs = options.playerConfigs ?? PLAYER_CONFIGS;
    const aiModes = options.aiModes ?? AI_MODES;
    let overlayFormatter = options.overlayFormatter ?? defaultOverlayFormatter;
    let afterDraw = options.afterDraw ?? null;
    let frameHook = options.onFrame ?? null;
    let collisionHandler = options.onPlayerCollision ?? null;
    const onResetKey = options.onResetKey ?? null;
    const enableDefaultResetKey = options.enableDefaultResetKey !== false;

    const players = createPlayers(playerConfigs);
    const keys = new Set();
    const controlKeys = new Set();
    playerConfigs.forEach((config) => {
      Object.values(config.controls).forEach((code) => controlKeys.add(code));
    });

    let showVelocityVector = !!options.showVelocityVector;
    let paused = !!options.startPaused;
    let running = false;
    let rafId = null;
    let lastTime = performance.now();
    let currentMap = maps.find((map) => map.id === options.initialMapId) ?? maps[0];
    let world = { ...currentMap.world };
    let obstacles = currentMap.obstacles;
    let killPlane = currentMap.killPlane ?? (currentMap.world?.floorLevel ?? -2) - 4;

    function respawnPlayer(player) {
      const spawnList = currentMap.spawnPoints ?? [];
      let spawn = null;
      if (spawnList.length > 0) {
        spawn = spawnList[player.index % spawnList.length] ?? spawnList[spawnList.length - 1];
      }
      if (!spawn) {
        spawn = { x: 0, y: 1.5 };
      }
      player.x = spawn.x;
      player.y = spawn.y;
      player.vx = 0;
      player.vy = 0;
      player.touchingGround = false;
      player.aiIntent = createBlankIntent();
    }

    function respawnAllPlayers() {
      players.forEach(respawnPlayer);
      settlePlayers();
    }

    function settlePlayers() {
      players.forEach((player) => {
        player.touchingGround = false;
        obstacles.forEach((rect) => resolveCircleRect(player, rect));
      });
      resolvePlayerCollisions();
    }

    function loadMap(mapId) {
      const nextMap = maps.find((map) => map.id === mapId) ?? maps[0];
      currentMap = nextMap;
      world = { ...nextMap.world };
      obstacles = nextMap.obstacles;
      killPlane = nextMap.killPlane ?? (nextMap.world?.floorLevel ?? -2) - 4;
      respawnAllPlayers();
      if (mapSelect) {
        mapSelect.value = currentMap?.id ?? "";
      }
    }

    function resolveCircleRect(ball, rect) {
      const closestX = Math.max(rect.x, Math.min(ball.x, rect.x + rect.w));
      const closestY = Math.max(rect.y, Math.min(ball.y, rect.y + rect.h));
      let dx = ball.x - closestX;
      let dy = ball.y - closestY;
      const distSq = dx * dx + dy * dy;
      const radiusSq = BALL_RADIUS * BALL_RADIUS;
      if (distSq >= radiusSq) {
        return;
      }

      let dist = Math.sqrt(distSq);
      if (dist === 0) {
        dist = 1e-6;
        dx = 0;
        dy = 1;
      }

      const overlap = BALL_RADIUS - dist;
      const nx = dx / dist;
      const ny = dy / dist;
      ball.x += nx * overlap;
      ball.y += ny * overlap;

      const vn = ball.vx * nx + ball.vy * ny;
      if (vn < 0) {
        ball.vx -= vn * nx;
        ball.vy -= vn * ny;
      }

      if (ny > 0.6) {
        const centerAbove =
          ball.x >= rect.x - 1e-6 && ball.x <= rect.x + rect.w + 1e-6;
        if (centerAbove) ball.touchingGround = true;
      }
    }

    function resolvePlayerPair(a, b) {
      let dx = b.x - a.x;
      let dy = b.y - a.y;
      const minDist = BALL_DIAMETER;
      let distSq = dx * dx + dy * dy;
      if (distSq === 0) {
        dx = 1e-6;
        dy = 0;
        distSq = dx * dx + dy * dy;
      }
      const dist = Math.sqrt(distSq);
      if (dist >= minDist) {
        return;
      }

      if (typeof collisionHandler === "function") {
        collisionHandler(a, b);
      }

      const overlap = minDist - dist;
      const nx = dx / dist;
      const ny = dy / dist;
      const correction = overlap / 2;
      a.x -= nx * correction;
      a.y -= ny * correction;
      b.x += nx * correction;
      b.y += ny * correction;

      const relVx = b.vx - a.vx;
      const relVy = b.vy - a.vy;
      const relVelAlongNormal = relVx * nx + relVy * ny;
      if (relVelAlongNormal > 0) {
        return;
      }

      const j = (-(1 + PLAYER_RESTITUTION) * relVelAlongNormal) / 2;
      a.vx -= j * nx;
      a.vy -= j * ny;
      b.vx += j * nx;
      b.vy += j * ny;
    }

    function resolvePlayerCollisions() {
      for (let i = 0; i < players.length; i++) {
        for (let j = i + 1; j < players.length; j++) {
          resolvePlayerPair(players[i], players[j]);
        }
      }
    }

    function computeSimpleXYIntent(player, target) {
      const intent = createBlankIntent();
      if (!target) return intent;
      const tol = 0.05;
      if (player.x < target.x - tol) intent.right = true;
      else if (player.x > target.x + tol) intent.left = true;
      if (player.y < target.y - tol) intent.up = true;
      else if (player.y > target.y + tol) intent.down = true;
      return intent;
    }

    function updateAIIntents() {
      const reference = players[0];
      players.forEach((player) => {
        player.aiIntent = createBlankIntent();
        if (player.aiMode === "simple-xy" && player !== reference && reference) {
          player.aiIntent = computeSimpleXYIntent(player, reference);
        }
      });
    }

    function physicsStep(dt) {
      players.forEach((player) => {
        const controls = player.controls;
        const intent = player.aiIntent ?? createBlankIntent();
        let ax = 0;
        if (keys.has(controls.left) || intent.left) ax -= H_THRUST;
        if (keys.has(controls.right) || intent.right) ax += H_THRUST;

        let ay = GRAVITY;
        if (keys.has(controls.up) || intent.up) ay += UP_THRUST;
        if (keys.has(controls.down) || intent.down) ay -= DOWN_THRUST;

        player.vx += ax * dt;
        player.vy += ay * dt;
        player.x += player.vx * dt;
        player.y += player.vy * dt;

        player.touchingGround = false;
        obstacles.forEach((rect) => resolveCircleRect(player, rect));

        if ((keys.has(controls.up) || intent.up) && player.touchingGround && player.vy <= 0) {
          player.vy = JUMP_SPEED;
          player.touchingGround = false;
        }

        player.x = Math.max(world.left + BALL_RADIUS, Math.min(player.x, world.right - BALL_RADIUS));
        if (player.x <= world.left + BALL_RADIUS && player.vx < 0) player.vx = 0;
        if (player.x >= world.right - BALL_RADIUS && player.vx > 0) player.vx = 0;

        const ceiling = world.ceiling - BALL_RADIUS;
        if (player.y > ceiling) {
          player.y = ceiling;
          if (player.vy > 0) player.vy = 0;
        }

        if (player.y < killPlane) {
          respawnPlayer(player);
        }
      });
      resolvePlayerCollisions();
    }

    function drawScene() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.save();
      ctx.translate(canvas.width / 2, canvas.height - 80);
      ctx.scale(1, -1);
      ctx.scale(PX_PER_M, PX_PER_M);

      ctx.lineWidth = 1 / PX_PER_M;
      ctx.strokeStyle = "rgba(94, 94, 121, 0.25)";
      ctx.beginPath();
      const gridSpacing = 1;
      const gridPadding = 2;
      const minX = Math.floor(world.left - gridPadding);
      const maxX = Math.ceil(world.right + gridPadding);
      const minY = Math.floor((world.floorLevel ?? -2) - gridPadding);
      const maxY = Math.ceil(world.ceiling + gridPadding);
      for (let x = minX; x <= maxX; x += gridSpacing) {
        ctx.moveTo(x, minY);
        ctx.lineTo(x, maxY);
      }
      for (let y = minY; y <= maxY; y += gridSpacing) {
        ctx.moveTo(minX, y);
        ctx.lineTo(maxX, y);
      }
      ctx.stroke();

      ctx.fillStyle = "#4b5563";
      obstacles.forEach((rect) => {
        ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
      });

      players.forEach((player) => {
        ctx.fillStyle = player.color;
        ctx.beginPath();
        ctx.arc(player.x, player.y, BALL_RADIUS, 0, Math.PI * 2);
        ctx.fill();

        if (showVelocityVector) {
          ctx.strokeStyle = "#f85149";
          ctx.lineWidth = 2 / PX_PER_M;
          ctx.beginPath();
          ctx.moveTo(player.x, player.y);
          ctx.lineTo(player.x + player.vx * 0.15, player.y + player.vy * 0.15);
          ctx.stroke();
        }
      });

      ctx.restore();

      if (overlay && overlayFormatter) {
        overlay.textContent = overlayFormatter(players, currentMap);
      }

      if (typeof afterDraw === "function") {
        afterDraw(ctx, {
          canvas,
          players,
          world,
          showVelocityVector,
          paused,
        });
      }
    }

    function handleKeydown(ev) {
      if (controlKeys.has(ev.code)) {
        ev.preventDefault();
      }
      keys.add(ev.code);
      if (ev.code === "KeyR") {
        if (typeof onResetKey === "function") {
          onResetKey();
        } else if (enableDefaultResetKey) {
          respawnAllPlayers();
        }
      }
    }

    function handleKeyup(ev) {
      keys.delete(ev.code);
    }

    function initUI() {
      if (resetButton) {
        resetButton.addEventListener("click", respawnAllPlayers);
      }

      if (velocityToggle) {
        velocityToggle.checked = showVelocityVector;
        velocityToggle.addEventListener("change", () => {
          showVelocityVector = velocityToggle.checked;
        });
      }

      if (mapSelect) {
        maps.forEach((map) => {
          const option = document.createElement("option");
          option.value = map.id;
          option.textContent = map.name;
          mapSelect.appendChild(option);
        });
        mapSelect.value = currentMap?.id ?? "";
        mapSelect.addEventListener("change", (event) => {
          loadMap(event.target.value);
        });
      }

      if (aiSelect && players[1]) {
        aiModes.forEach((mode) => {
          const option = document.createElement("option");
          option.value = mode.id;
          option.textContent = mode.name;
          aiSelect.appendChild(option);
        });
        aiSelect.value = players[1].aiMode;
        aiSelect.addEventListener("change", (event) => {
          players[1].aiMode = event.target.value;
        });
      }

      window.addEventListener("keydown", handleKeydown);
      window.addEventListener("keyup", handleKeyup);
    }

    function loop(now) {
      const dt = (now - lastTime) / 1000;
      lastTime = now;
      const clampedDt = Math.min(dt, 1 / 30);
      if (!paused) {
        updateAIIntents();
        physicsStep(clampedDt);
      }
      drawScene();
      if (typeof frameHook === "function") {
        frameHook({
          dt: clampedDt,
          now,
          paused,
          players,
          currentMap,
        });
      }
      rafId = requestAnimationFrame(loop);
    }

    function start() {
      if (!running) {
        running = true;
        lastTime = performance.now();
        rafId = requestAnimationFrame(loop);
      }
    }

    function stop() {
      if (running) {
        cancelAnimationFrame(rafId);
        running = false;
      }
    }

    function setPlayerAIMode(index, modeId) {
      if (!players[index]) return;
      players[index].aiMode = modeId;
      if (aiSelect && index === 1) {
        aiSelect.value = modeId;
      }
    }

    initUI();
    loadMap(currentMap?.id);

    const api = {
      start,
      stop,
      pause: () => {
        paused = true;
      },
      resume: () => {
        paused = false;
      },
      isPaused: () => paused,
      respawnPlayers: respawnAllPlayers,
      setMap: loadMap,
      setPlayerAIMode,
      setVelocityVectorVisible: (value) => {
        showVelocityVector = !!value;
        if (velocityToggle) {
          velocityToggle.checked = showVelocityVector;
        }
      },
      setOverlayFormatter: (fn) => {
        overlayFormatter = fn;
      },
      setAfterDraw: (fn) => {
        afterDraw = fn;
      },
      setFrameHook: (fn) => {
        frameHook = fn;
      },
      setCollisionHandler: (fn) => {
        collisionHandler = fn;
      },
      getPlayers: () => players,
      getCurrentMap: () => currentMap,
      setPaused: (value) => {
        paused = !!value;
      },
    };

    return api;
  }

  global.BonkSandbox = {
    createSandbox,
    MAPS,
    AI_MODES,
    PLAYER_CONFIGS,
  };
})(window);
