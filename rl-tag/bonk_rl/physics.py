from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Dict, List, Sequence, Tuple


BALL_DIAMETER = 1.0
BALL_RADIUS = BALL_DIAMETER / 2.0

ACTION_TO_INTENT = {
    0: (False, False, False, False),
    1: (True, False, False, False),
    2: (False, True, False, False),
    3: (False, False, True, False),
    4: (False, False, False, True),
    5: (True, False, True, False),
    6: (False, True, True, False),
    7: (True, False, False, True),
    8: (False, True, False, True),
}


@dataclass
class PlayerState:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    touching_ground: bool = False


@dataclass
class SimState:
    players: List[PlayerState] = field(default_factory=lambda: [PlayerState(), PlayerState()])
    done: bool = False
    caught: bool = False
    steps: int = 0


@dataclass
class PhysicsConfig:
    gravity: float = -9.5
    up_thrust: float = 5.7
    down_thrust: float = 5.7
    h_thrust: float = 5.45
    tap_apex: float = 1.2
    restitution: float = 0.4
    dt: float = 1.0 / 60.0

    @property
    def jump_speed(self) -> float:
        return math.sqrt(2.0 * abs(self.gravity) * self.tap_apex)


BASE_MAPS: List[Dict] = [
    {
        "id": "flat-1v1",
        "world": {"floorLevel": -2.0, "ceiling": 9.2, "left": -11.5, "right": 11.5},
        "obstacles": [{"x": -5.0, "y": 2.8, "w": 10.0, "h": 0.35}],
        "spawnPoints": [{"x": -3.8, "y": 3.2}, {"x": 3.8, "y": 3.2}],
        "killPlane": -4.5,
    },
    {
        "id": "flat-wide",
        "world": {"floorLevel": -2.0, "ceiling": 9.2, "left": -11.5, "right": 11.5},
        "obstacles": [{"x": -11.5, "y": 2.8, "w": 23.0, "h": 0.35}],
        "spawnPoints": [{"x": -8.5, "y": 3.2}, {"x": 8.5, "y": 3.2}],
        "killPlane": -5.0,
    },
    {
        "id": "pyramid",
        "world": {"floorLevel": -2.0, "ceiling": 9.2, "left": -12.0, "right": 12.0},
        "obstacles": [
            {"x": -10.0, "y": 1.0, "w": 20.0, "h": 0.5},
            {"x": -7.5, "y": 2.0, "w": 15.0, "h": 0.5},
            {"x": -5.0, "y": 3.0, "w": 10.0, "h": 0.5},
            {"x": -2.5, "y": 4.0, "w": 5.0, "h": 0.5},
        ],
        "spawnPoints": [{"x": -9.0, "y": 1.5}, {"x": 9.0, "y": 1.5}],
        "killPlane": -5.0,
    },
]


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class TagPhysics:
    def __init__(
        self,
        max_steps: int = 900,
        domain_randomization: bool = True,
        rng: random.Random | None = None,
    ) -> None:
        self.max_steps = max_steps
        self.domain_randomization = domain_randomization
        self.rng = rng or random.Random()
        self.base_config = PhysicsConfig()
        self.config = PhysicsConfig()
        self.world: Dict = {}
        self.obstacles: List[Dict] = []
        self.spawn_points: List[Dict] = []
        self.kill_plane: float = -5.0
        self.state = SimState()

    def reset(self, seed: int | None = None) -> SimState:
        if seed is not None:
            self.rng.seed(seed)
        self._sample_map_and_physics()
        self.state = SimState()
        for i in range(2):
            self._respawn_player(self.state.players[i], i)
        self._resolve_player_collisions()
        self.state.steps = 0
        self.state.done = False
        self.state.caught = False
        return self.state

    def step(self, catcher_action: int, evader_action: int) -> Tuple[SimState, Dict]:
        if self.state.done:
            return self.state, {}

        actions = [catcher_action, evader_action]
        for idx, player in enumerate(self.state.players):
            left, right, up, down = ACTION_TO_INTENT.get(actions[idx], ACTION_TO_INTENT[0])
            ax = 0.0
            if left:
                ax -= self.config.h_thrust
            if right:
                ax += self.config.h_thrust

            ay = self.config.gravity
            if up:
                ay += self.config.up_thrust
            if down:
                ay -= self.config.down_thrust

            player.vx += ax * self.config.dt
            player.vy += ay * self.config.dt
            player.x += player.vx * self.config.dt
            player.y += player.vy * self.config.dt

            player.touching_ground = False
            for rect in self.obstacles:
                self._resolve_circle_rect(player, rect)

            if up and player.touching_ground and player.vy <= 0.0:
                player.vy = self.config.jump_speed
                player.touching_ground = False

            player.x = _clip(player.x, self.world["left"] + BALL_RADIUS, self.world["right"] - BALL_RADIUS)
            if player.x <= self.world["left"] + BALL_RADIUS and player.vx < 0.0:
                player.vx = 0.0
            if player.x >= self.world["right"] - BALL_RADIUS and player.vx > 0.0:
                player.vx = 0.0

            ceiling = self.world["ceiling"] - BALL_RADIUS
            if player.y > ceiling:
                player.y = ceiling
                if player.vy > 0.0:
                    player.vy = 0.0

            if player.y < self.kill_plane:
                self._respawn_player(player, idx)

        caught_now = self._resolve_player_collisions()
        self.state.steps += 1
        self.state.caught = caught_now
        self.state.done = caught_now or self.state.steps >= self.max_steps
        info = {
            "caught": caught_now,
            "timeout": self.state.steps >= self.max_steps and not caught_now,
            "map_id": self.world.get("id", "unknown"),
        }
        return self.state, info

    def _sample_map_and_physics(self) -> None:
        base_map = self.rng.choice(BASE_MAPS)
        self.world = dict(base_map["world"])
        self.world["id"] = base_map["id"]
        self.obstacles = [dict(o) for o in base_map["obstacles"]]
        self.spawn_points = [dict(s) for s in base_map["spawnPoints"]]
        self.kill_plane = float(base_map["killPlane"])

        self.config = PhysicsConfig(**self.base_config.__dict__)
        if not self.domain_randomization:
            return

        gravity_scale = self.rng.uniform(0.95, 1.05)
        thrust_scale = self.rng.uniform(0.95, 1.05)
        restitution_scale = self.rng.uniform(0.9, 1.1)
        spawn_jitter = 0.45

        self.config.gravity *= gravity_scale
        self.config.up_thrust *= thrust_scale
        self.config.down_thrust *= thrust_scale
        self.config.h_thrust *= thrust_scale
        self.config.restitution = _clip(self.config.restitution * restitution_scale, 0.2, 0.8)

        for rect in self.obstacles:
            rect["x"] += self.rng.uniform(-0.25, 0.25)
            rect["y"] += self.rng.uniform(-0.15, 0.15)
            rect["w"] = max(0.2, rect["w"] * self.rng.uniform(0.9, 1.1))
            rect["h"] = max(0.15, rect["h"] * self.rng.uniform(0.9, 1.1))

        for spawn in self.spawn_points:
            spawn["x"] += self.rng.uniform(-spawn_jitter, spawn_jitter)
            spawn["y"] += self.rng.uniform(-spawn_jitter * 0.5, spawn_jitter * 0.5)

    def _respawn_player(self, player: PlayerState, index: int) -> None:
        if self.spawn_points:
            spawn = self.spawn_points[index % len(self.spawn_points)]
        else:
            spawn = {"x": 0.0, "y": 1.5}
        player.x = float(spawn["x"])
        player.y = float(spawn["y"])
        player.vx = 0.0
        player.vy = 0.0
        player.touching_ground = False

    def _resolve_circle_rect(self, ball: PlayerState, rect: Dict) -> None:
        closest_x = _clip(ball.x, rect["x"], rect["x"] + rect["w"])
        closest_y = _clip(ball.y, rect["y"], rect["y"] + rect["h"])

        dx = ball.x - closest_x
        dy = ball.y - closest_y
        dist_sq = dx * dx + dy * dy
        radius_sq = BALL_RADIUS * BALL_RADIUS
        if dist_sq >= radius_sq:
            return

        dist = math.sqrt(dist_sq) if dist_sq > 0.0 else 1e-6
        if dist_sq == 0.0:
            dx = 0.0
            dy = 1.0

        overlap = BALL_RADIUS - dist
        nx = dx / dist
        ny = dy / dist
        ball.x += nx * overlap
        ball.y += ny * overlap

        vn = ball.vx * nx + ball.vy * ny
        if vn < 0.0:
            ball.vx -= vn * nx
            ball.vy -= vn * ny

        if ny > 0.6:
            center_above = rect["x"] - 1e-6 <= ball.x <= rect["x"] + rect["w"] + 1e-6
            if center_above:
                ball.touching_ground = True

    def _resolve_player_collisions(self) -> bool:
        a = self.state.players[0]
        b = self.state.players[1]
        dx = b.x - a.x
        dy = b.y - a.y
        min_dist = BALL_DIAMETER
        dist_sq = dx * dx + dy * dy
        if dist_sq == 0.0:
            dx = 1e-6
            dy = 0.0
            dist_sq = dx * dx + dy * dy
        dist = math.sqrt(dist_sq)
        if dist >= min_dist:
            return False

        overlap = min_dist - dist
        nx = dx / dist
        ny = dy / dist
        correction = overlap / 2.0
        a.x -= nx * correction
        a.y -= ny * correction
        b.x += nx * correction
        b.y += ny * correction

        rel_vx = b.vx - a.vx
        rel_vy = b.vy - a.vy
        rel_vel_along_normal = rel_vx * nx + rel_vy * ny
        if rel_vel_along_normal <= 0.0:
            j = (-(1.0 + self.config.restitution) * rel_vel_along_normal) / 2.0
            a.vx -= j * nx
            a.vy -= j * ny
            b.vx += j * nx
            b.vy += j * ny
        return True


def build_obs(state: SimState, world: Dict, self_index: int) -> List[float]:
    me = state.players[self_index]
    other = state.players[1 - self_index]
    width = world["right"] - world["left"]
    height = world["ceiling"] - world["floorLevel"]

    return [
        me.x / max(width, 1e-6),
        me.y / max(height, 1e-6),
        me.vx / 20.0,
        me.vy / 20.0,
        other.x / max(width, 1e-6),
        other.y / max(height, 1e-6),
        other.vx / 20.0,
        other.vy / 20.0,
        (other.x - me.x) / max(width, 1e-6),
        (other.y - me.y) / max(height, 1e-6),
        (other.vx - me.vx) / 20.0,
        (other.vy - me.vy) / 20.0,
        float(me.touching_ground),
        float(other.touching_ground),
        float(state.steps) / 900.0,
        1.0,
    ]
