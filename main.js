const canvas = document.getElementById('game');
const ctx = canvas.getContext('2d');

const WORLD = {
  width: canvas.width,
  height: canvas.height,
  floor: canvas.height - 70,
  gravity: 2200,
  airDrag: 0.992,
  restitutionWalls: 0.35,
  restitutionBonk: 0.92,
};

const SURFACES = [
  {
    x1: 0,
    x2: canvas.width,
    y: WORLD.floor,
    type: 'bounce',
    bounce: 0.94,
    bounceThreshold: 240,
    minBounceVelocity: 460,
    upBoost: 1.12,
    friction: 0,
  },
  {
    x1: WORLD.width * 0.35,
    x2: WORLD.width * 0.65,
    y: WORLD.floor - 120,
    type: 'flat',
    friction: 1,
  },
];

const controls = new Set();
window.addEventListener('keydown', (event) => {
  if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' '].includes(event.key)) {
    event.preventDefault();
  }
  controls.add(event.code);
});
window.addEventListener('keyup', (event) => {
  controls.delete(event.code);
});

class Bonk {
  constructor(options) {
    this.x = options.x;
    this.y = options.y;
    this.prevX = this.x;
    this.prevY = this.y;
    this.vx = options.vx ?? 0;
    this.vy = options.vy ?? 0;
    this.color = options.color ?? '#4cc9f0';
    this.outline = options.outline ?? '#e5e7eb';
    this.baseRadius = options.radius ?? 26;
    this.radius = this.baseRadius;
    this.baseMass = options.mass ?? 1;
    this.heavyMass = options.heavyMass ?? this.baseMass * 4;
    this.mass = this.baseMass;
    this.control = options.control ?? 'none';
    this.isHeavy = false;
    this.grounded = false;
    this.timeSinceGrounded = 0;
    this.jumpImpulse = options.jumpImpulse ?? 780;
    this.moveForce = options.moveForce ?? 1650;
    this.heavyMoveForce = options.heavyMoveForce ?? 1100;
    this.fastFallForce = options.fastFallForce ?? 1400;
    this.maxAirSpeed = options.maxAirSpeed ?? 420;
    this.maxGroundSpeed = options.maxGroundSpeed ?? 520;
    this.jumpGrace = options.jumpGrace ?? 0.18;
    this.forceX = 0;
    this.forceY = 0;
  }

  resetForces() {
    this.forceX = 0;
    this.forceY = 0;
  }

  setHeavy(active) {
    this.isHeavy = active;
    this.mass = active ? this.heavyMass : this.baseMass;
    this.radius = active ? this.baseRadius * 0.95 : this.baseRadius;
  }

  canJump() {
    return this.timeSinceGrounded < this.jumpGrace;
  }

  applyControls(dt, now) {
    if (this.control === 'player') {
      const heavyActive = controls.has('KeyX');
      this.setHeavy(heavyActive);

      const moveForce = this.isHeavy ? this.heavyMoveForce : this.moveForce;
      if (controls.has('ArrowLeft')) this.forceX -= moveForce;
      if (controls.has('ArrowRight')) this.forceX += moveForce;

      const wantsJump = controls.has('ArrowUp') || controls.has('Space');
      if (wantsJump && this.canJump()) {
        this.vy = -this.jumpImpulse;
        this.grounded = false;
        this.timeSinceGrounded = Number.POSITIVE_INFINITY;
      }

      if (controls.has('ArrowDown') && !this.grounded) {
        this.forceY += this.fastFallForce;
      }
    } else if (this.control === 'dummy') {
      // Lightweight movement so there are regular collisions.
      const heavyWindow = Math.sin(now * 0.0008) > 0.85;
      this.setHeavy(heavyWindow);
      const targetX = WORLD.width * 0.55 + Math.sin(now * 0.0006) * 180;
      const dir = Math.sign(targetX - this.x) || 0;
      const targetForce = (this.isHeavy ? 800 : 1200) * dir;
      this.forceX += targetForce;

      if (this.grounded && Math.random() < 0.005) {
        this.vy = -600 - Math.random() * 200;
        this.grounded = false;
        this.timeSinceGrounded = Number.POSITIVE_INFINITY;
      }
    }
  }

  integrate(dt) {
    const groundedLastFrame = this.grounded;
    this.prevX = this.x;
    this.prevY = this.y;
    const ax = this.forceX / this.mass;
    const ay = WORLD.gravity + this.forceY / this.mass;

    this.vx += ax * dt;
    this.vy += ay * dt;

    if (!groundedLastFrame) {
      this.vx *= WORLD.airDrag;
      this.vy *= WORLD.airDrag;
    } else {
      this.vy *= WORLD.airDrag;
    }

    this.x += this.vx * dt;
    this.y += this.vy * dt;

    this.timeSinceGrounded += dt;
    this.grounded = false;
  }

  applySurfaceConstraints(surfaces) {
    const downPress = this.control === 'player' && controls.has('ArrowDown');
    const upPress =
      this.control === 'player' &&
      (controls.has('ArrowUp') || controls.has('Space'));

    for (const surface of surfaces) {
      const withinX =
        this.x + this.radius > surface.x1 && this.x - this.radius < surface.x2;
      const crossingDownward =
        this.prevY + this.radius <= surface.y &&
        this.y + this.radius >= surface.y &&
        this.vy >= 0;
      if (!withinX || !crossingDownward) continue;

      this.y = surface.y - this.radius;
      const incomingVy = Math.max(this.vy, 0);
      if (surface.type === 'bounce') {
        const threshold = surface.bounceThreshold ?? 200;
        const wantsBounce = incomingVy > threshold || downPress;
        if (wantsBounce) {
          const minVelocity = surface.minBounceVelocity ?? 420;
          const bounceFactor = surface.bounce ?? 0.9;
          const baseVelocity = downPress
            ? Math.max(incomingVy * 1.1, minVelocity)
            : incomingVy;
          let newVy = -baseVelocity * bounceFactor;
          if (upPress) newVy *= surface.upBoost ?? 1.08;
          this.vy = newVy;
          this.grounded = false;
          this.timeSinceGrounded = Number.POSITIVE_INFINITY;
        } else {
          this.vy = 0;
          this.grounded = true;
          this.timeSinceGrounded = 0;
        }
      } else {
        this.vy = 0;
        this.grounded = true;
        this.timeSinceGrounded = 0;
      }

      if (surface.friction) {
        this.vx *= Math.max(0, 1 - surface.friction);
      }

      return;
    }
  }

  clampHorizontalVelocity() {
    const limit = this.grounded ? this.maxGroundSpeed : this.maxAirSpeed;
    if (Math.abs(this.vx) > limit) {
      this.vx = limit * Math.sign(this.vx);
    }
  }

  handleBounds() {
    // Left wall
    if (this.x - this.radius < 0) {
      this.x = this.radius;
      if (this.vx < 0) this.vx = -this.vx * WORLD.restitutionWalls;
    }

    // Right wall
    if (this.x + this.radius > WORLD.width) {
      this.x = WORLD.width - this.radius;
      if (this.vx > 0) this.vx = -this.vx * WORLD.restitutionWalls;
    }

    // Ceiling
    if (this.y - this.radius < 0) {
      this.y = this.radius;
      if (this.vy < 0) this.vy = -this.vy * 0.4;
    }
  }
}

const player = new Bonk({
  x: WORLD.width * 0.25,
  y: WORLD.floor - 40,
  color: '#4cc9f0',
  outline: '#1fb6ff',
  control: 'player',
});

const dummy = new Bonk({
  x: WORLD.width * 0.65,
  y: WORLD.floor - 40,
  color: '#fb8500',
  outline: '#ffd166',
  control: 'dummy',
  baseMass: 1.2,
  moveForce: 1500,
  heavyMoveForce: 1000,
  maxAirSpeed: 360,
  maxGroundSpeed: 420,
});

dummy.vx = -120;
const bonks = [player, dummy];

function resolveBonkCollision(a, b) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const dist = Math.hypot(dx, dy);
  const minDist = a.radius + b.radius;
  if (dist === 0) return;
  if (dist >= minDist) return;

  const nx = dx / dist;
  const ny = dy / dist;
  const penetration = minDist - dist;
  const totalMass = a.mass + b.mass;
  const ratioA = b.mass / totalMass;
  const ratioB = a.mass / totalMass;

  // Positional correction keeps the circles from sticking.
  a.x -= nx * penetration * ratioA;
  a.y -= ny * penetration * ratioA;
  b.x += nx * penetration * ratioB;
  b.y += ny * penetration * ratioB;

  const relVx = b.vx - a.vx;
  const relVy = b.vy - a.vy;
  const velAlongNormal = relVx * nx + relVy * ny;

  if (velAlongNormal > 0) return;

  const restitution = WORLD.restitutionBonk;
  const impulse =
    (-(1 + restitution) * velAlongNormal) / (1 / a.mass + 1 / b.mass);
  const impulseX = impulse * nx;
  const impulseY = impulse * ny;

  a.vx -= impulseX / a.mass;
  a.vy -= impulseY / a.mass;
  b.vx += impulseX / b.mass;
  b.vy += impulseY / b.mass;
}

const heavyStateEl = document.getElementById('heavyState');
const playerSpeedEl = document.getElementById('playerSpeed');
const dummySpeedEl = document.getElementById('dummySpeed');

function drawFloor() {
  ctx.fillStyle = '#111521';
  ctx.fillRect(0, 0, WORLD.width, WORLD.height);
  ctx.fillStyle = '#0c1320';
  ctx.fillRect(0, WORLD.floor, WORLD.width, WORLD.height - WORLD.floor);
  for (const surface of SURFACES) {
    const height = surface.type === 'flat' ? 18 : 12;
    ctx.fillStyle = surface.type === 'flat' ? '#6d28d9' : '#1d4ed8';
    ctx.fillRect(surface.x1, surface.y - height, surface.x2 - surface.x1, height);
  }
}

function drawBonk(b) {
  ctx.beginPath();
  ctx.arc(b.x, b.y, b.radius, 0, Math.PI * 2);
  ctx.fillStyle = b.isHeavy ? '#ffffff' : b.color;
  ctx.fill();
  ctx.lineWidth = b.isHeavy ? 5 : 3;
  ctx.strokeStyle = b.outline;
  ctx.stroke();

  // Draw a simple velocity indicator.
  ctx.beginPath();
  ctx.moveTo(b.x, b.y);
  ctx.lineTo(b.x + b.vx * 0.05, b.y + b.vy * 0.05);
  ctx.strokeStyle = 'rgba(255,255,255,0.35)';
  ctx.lineWidth = 2;
  ctx.stroke();
}

let lastTime = performance.now();

function frame(now) {
  const dt = Math.min((now - lastTime) / 1000, 0.032);
  lastTime = now;

  for (const bonk of bonks) {
    bonk.resetForces();
    bonk.applyControls(dt, now);
  }

  for (const bonk of bonks) {
    bonk.integrate(dt);
    bonk.applySurfaceConstraints(SURFACES);
    bonk.handleBounds();
    bonk.clampHorizontalVelocity();
  }

  resolveBonkCollision(player, dummy);

  ctx.clearRect(0, 0, WORLD.width, WORLD.height);
  drawFloor();
  for (const bonk of bonks) {
    drawBonk(bonk);
  }

  updatePanel();
  requestAnimationFrame(frame);
}

function updatePanel() {
  heavyStateEl.textContent = player.isHeavy ? 'yes' : 'no';
  const playerSpeed = Math.hypot(player.vx, player.vy);
  const dummySpeed = Math.hypot(dummy.vx, dummy.vy);
  playerSpeedEl.textContent = playerSpeed.toFixed(1);
  dummySpeedEl.textContent = dummySpeed.toFixed(1);
}

requestAnimationFrame(frame);
