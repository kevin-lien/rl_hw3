"""Astro Party game logic: the arena, physics, weapons, and observations.
DON'T MODIFY: the TAs grade with the original version of this file.

Shared by both environments, so training and evaluation play exactly the same
game. Nothing in here is a reward. Each step returns a list of `Event`s (what
happened this tick), and the environments turn those into rewards:
  - envs/train_astro_party_env.py: your dense training reward (edit that file),
  - envs/eval_astro_party_env.py:  the sparse grading reward (+1/-1 per final kill).

The game (Astro Party "Pilot Hunters" mode): two ships drift around a walled
arena with no gravity, bounce off walls, the planet, cover structures and each
other, shoot a limited-ammo gun, and can pick up a one-shot laser that drops
from destroyed asteroids. Destroying a ship ejects its pilot as a small,
gun-less astronaut; only killing *that* (shooting it, or ramming it with your
ship) is the final elimination that ends the duel. Survive as a pilot for
RESPAWN_STEPS and you're re-issued a ship. Each game starts from a random
(seeded) spawn: ship1 at a random safe spot, ship2 mirrored through the planet,
at least MIN_SPAWN_DIST apart.

Action (per ship), the native form used inside the game:
  {"move": [turn_rate, thrust] in [-1, 1], "shoot": 0 or 1}
Agents use the flat Box(3,) [turn_rate, thrust, shoot_signal] form instead;
decode_action() converts it (shoot_signal > 0 means "fire"). turn_rate/thrust
are relative to the current heading, the same whether piloting a ship or an
ejected astronaut. Shooting is ignored while ejected.

Observation (Box(84,)), fully observable, self-centric: the same layout from
either ship's point of view, so one policy can fly either ship. Positions are
normalized by half the arena size (x / 600, y / 400), ship velocities by
MAX_SPEED, bullet velocities by BULLET_SPEED. Turning has no inertia (the action
sets the turn rate directly), so there is no angular-velocity entry.
  own ship (14):
     0- 1  x, y               position (arena centre = 0)
     2- 3  vx, vy             velocity
     4- 5  cos(angle), sin(angle)   heading; the nose points along (sin, -cos)
     6     dist_to_planet     distance to the planet centre, / 600
     7     dist_to_wall       distance to the nearest cover bar, / 600
     8     ammo_frac          rounds in the magazine / MAX_AMMO (3)
     9     reload_frac        progress towards the next round (1.0 when full)
    10     has_laser          1 if holding a laser pickup
    11     ready              1 if the fire cooldown is over (it lasts FIRE_COOLDOWN_STEPS
                              after each shot); firing also needs ammo (or a laser) and
                              fewer than BULLET_SLOTS bullets in flight
    12     is_astronaut       1 while ejected (can't shoot)
    13     respawn_frac       fraction of the respawn wait left while ejected, else 0
  opponent, relative to you (12):
    14-15  dx, dy             opponent position minus yours
    16-17  dvx, dvy           opponent velocity minus yours
    18-19  cos, sin of (opponent angle - your angle)
    20     opp_alive          0 once the opponent is eliminated
    21     opp_is_astronaut
    22     opp_ammo_frac
    23     opp_reload_frac
    24     opp_has_laser
    25     opp_ready
  asteroids (26-37): 4 x (dx, dy, radius / 600), relative to you, fixed slot order
  pickups (38-43): 2 x (dx, dy, active), fixed slot order
  own bullets (44-63): 4 x (dx, dy, vx, vy, active), nearest first
  opponent bullets (64-83): same layout, for the opponent's bullets
Empty pickup/bullet slots are all zeros (active = 0).
"""
import math
from dataclasses import dataclass, field

import numpy as np
from gymnasium import spaces

FPS = 50
DT = 1.0 / FPS
MAX_STEPS = 1500

ARENA_W, ARENA_H = 1200.0, 800.0

SHIP_RADIUS = 16.0
SHIP_POLY_LOCAL = np.array([(0, -20), (-13, 14), (-6, 8), (6, 8), (13, 14)], dtype=np.float32)
MAX_SPEED = 260.0
THRUST_ACCEL = 700.0
REVERSE_ACCEL_SCALE = 0.5  # reverse/brake thrust is weaker than forward, like a real engine
MAX_TURN_RATE = 3.5  # rad/s
LINEAR_DRAG = 0.999  # per-step velocity decay; ships coast like they're in space, but not forever
WALL_RESTITUTION = 0.65
BODY_RESTITUTION = 0.75  # ship<->ship, ship<->asteroid, asteroid<->asteroid bounces

PLANET_POS = np.array([ARENA_W / 2, ARENA_H / 2], dtype=np.float32)
PLANET_RADIUS = 100.0
PLANET_RESTITUTION = 0.8

# Cover structures: each is a "#"-shaped cluster of solid bars (outer frame + a
# cross through the middle). Physical -- ships and asteroids bounce off the
# bars, and bullets/lasers are blocked by them, so a ship can duck into one of
# the four open pockets to break line of sight.
BRACKET_CENTERS = [
    (140, 130), (140, ARENA_H - 130),
    (ARENA_W - 140, 130), (ARENA_W - 140, ARENA_H - 130),
    (ARENA_W / 2, 40), (ARENA_W / 2, ARENA_H - 40),
]
BRACKET_W, BRACKET_H = 100.0, 66.0
WALL_BAR_THICK = 10.0


def _build_walls():
    walls = []
    hw, hh, t = BRACKET_W / 2, BRACKET_H / 2, WALL_BAR_THICK / 2
    for cx, cy in BRACKET_CENTERS:
        walls.append((cx, cy - hh, hw, t))  # top
        walls.append((cx, cy + hh, hw, t))  # bottom
        walls.append((cx - hw, cy, t, hh))  # left
        walls.append((cx + hw, cy, t, hh))  # right
        walls.append((cx, cy, hw, t))  # mid horizontal
        walls.append((cx, cy, t, hh))  # mid vertical
    return walls


WALLS = _build_walls()  # list of (center_x, center_y, half_w, half_h), fixed for the arena's lifetime

NUM_ASTEROIDS = 4
ASTEROID_RADIUS_RANGE = (18.0, 34.0)
ASTEROID_SPEED_RANGE = (10.0, 45.0)
ASTEROID_HP = 2  # normal-gun hits needed to destroy; a laser always destroys in one hit

BULLET_SPEED = 620.0
BULLET_RADIUS = 4.0
BULLET_TTL_STEPS = int(1.3 * FPS)
MAX_AMMO = 3  # magazine size: total rounds held, independent of how many are still in flight
FIRE_COOLDOWN_STEPS = 6  # min steps between shots -- how fast the mag can be dumped
RELOAD_STEPS_PER_AMMO = FPS  # steps to regenerate a single round -- much slower than firing,
                             # so dumping the whole magazine costs real downtime, not a blip
BULLET_SLOTS = 4  # hard per-ship cap on simultaneous in-flight bullets, so the observation's
                   # fixed bullet slots (see AstroPartyGame.observe) can never be truncated.

PICKUP_SLOTS = 2
PICKUP_RADIUS = 14.0
PICKUP_TTL_STEPS = int(8.0 * FPS)
PICKUP_DROP_PROB = 0.5

ASTRONAUT_RADIUS = 9.0
ASTRONAUT_POLY_LOCAL = np.array([(0, -9), (-6, 6), (-3, 4), (3, 4), (6, 6)], dtype=np.float32)
ASTRONAUT_TURN_RATE = MAX_TURN_RATE
ASTRONAUT_MAX_SPEED = 110.0
ASTRONAUT_ACCEL = 380.0
ASTRONAUT_DRAG = 0.995
RESPAWN_STEPS = int(4.0 * FPS)  # survive this long as a pilot and you're re-issued a ship

# Spawns: ship1 gets a random safe spot, ship2 the point mirrored through the planet
# (the map is point-symmetric), with the mirrored heading, so neither ship is favoured.
MIN_SPAWN_DIST = 500.0  # the two ships never start closer than this
SPAWN_CLEARANCE = 30.0  # extra margin around the planet, cover and arena edges

SHIP_IDS = ("ship1", "ship2")


def other(ship_id: str) -> str:
    return "ship2" if ship_id == "ship1" else "ship1"


# ------------------------------------------------------------------ events

@dataclass(frozen=True)
class Event:
    """Something that happened during one game step. `by` is the ship that did
    it; `target` is the ship it happened to (only for "eject" and "kill").

      kind            by                  target
      "eject"         shooter             ship that was destroyed (its pilot ejects)
      "kill"          shooter / rammer    astronaut that was killed (final elimination, ends the duel)
      "asteroid_hit"  shooter             -
      "pickup"        ship that grabbed a laser pickup
      "respawn"       pilot that survived long enough to get a new ship
    """
    kind: str
    by: str
    target: str | None = None


# ------------------------------------------------------------------ geometry

def _radius(ship) -> float:
    return SHIP_RADIUS if ship.mode == "ship" else ASTRONAUT_RADIUS


def _rotate(local_pts: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return local_pts @ rot.T


def _heading(angle: float) -> np.ndarray:
    return np.array([math.sin(angle), -math.cos(angle)], dtype=np.float32)


def _wall_bounce(pos: np.ndarray, vel: np.ndarray, r: float) -> None:
    if pos[0] - r < 0:
        pos[0] = r
        vel[0] = -vel[0] * WALL_RESTITUTION
    elif pos[0] + r > ARENA_W:
        pos[0] = ARENA_W - r
        vel[0] = -vel[0] * WALL_RESTITUTION
    if pos[1] - r < 0:
        pos[1] = r
        vel[1] = -vel[1] * WALL_RESTITUTION
    elif pos[1] + r > ARENA_H:
        pos[1] = ARENA_H - r
        vel[1] = -vel[1] * WALL_RESTITUTION


def _resolve_circle_collision(p1, v1, inv_m1, r1, p2, v2, inv_m2, r2, restitution):
    """Elastic collision + overlap correction between two circles (inv_m=0 == immovable)."""
    delta = p2 - p1
    dist = float(np.hypot(delta[0], delta[1]))
    min_dist = r1 + r2
    if dist < 1e-6:
        delta = np.array([1.0, 0.0], dtype=np.float32)
        dist = 1e-6
    if dist >= min_dist:
        return
    normal = delta / dist
    overlap = min_dist - dist
    total_inv = inv_m1 + inv_m2
    if total_inv <= 0:
        return
    p1 -= normal * overlap * (inv_m1 / total_inv)
    p2 += normal * overlap * (inv_m2 / total_inv)
    rel_vel = v1 - v2
    vel_along_normal = float(np.dot(rel_vel, normal))
    if vel_along_normal > 0:
        return
    j = -(1 + restitution) * vel_along_normal / total_inv
    impulse = j * normal
    v1 += impulse * inv_m1
    v2 -= impulse * inv_m2


def _ray_circle_t(origin, direction, center, radius):
    """Nearest t>=0 where the unit ray hits the circle, or None."""
    oc = origin - center
    b = 2 * float(np.dot(direction, oc))
    c = float(np.dot(oc, oc)) - radius * radius
    disc = b * b - 4 * c
    if disc < 0:
        return None
    sq = math.sqrt(disc)
    t1, t2 = (-b - sq) / 2, (-b + sq) / 2
    if t1 >= 0:
        return t1
    if t2 >= 0:
        return t2
    return None


def _circle_rect_collide(pos, vel, radius, cx, cy, hw, hh, restitution):
    """Bounce a moving circle off an axis-aligned rect (in place on pos/vel)."""
    closest_x = min(max(pos[0], cx - hw), cx + hw)
    closest_y = min(max(pos[1], cy - hh), cy + hh)
    dx, dy = pos[0] - closest_x, pos[1] - closest_y
    dist = math.hypot(dx, dy)
    if dist > 1e-6:
        if dist >= radius:
            return
        nx, ny = dx / dist, dy / dist
        overlap = radius - dist
    else:
        px, py = hw - abs(pos[0] - cx), hh - abs(pos[1] - cy)
        if px < py:
            nx, ny, overlap = (1.0 if pos[0] >= cx else -1.0), 0.0, px + radius
        else:
            nx, ny, overlap = 0.0, (1.0 if pos[1] >= cy else -1.0), py + radius
    pos[0] += nx * overlap
    pos[1] += ny * overlap
    vn = vel[0] * nx + vel[1] * ny
    if vn < 0:
        j = -(1 + restitution) * vn
        vel[0] += j * nx
        vel[1] += j * ny


def _ray_aabb_entry_t(origin, direction, cx, cy, hw, hh):
    """t>=0 where the ray first enters the rect, or None if it misses/is behind."""
    tmin, tmax = -math.inf, math.inf
    for o, d, lo, hi in [
        (origin[0], direction[0], cx - hw, cx + hw),
        (origin[1], direction[1], cy - hh, cy + hh),
    ]:
        if abs(d) < 1e-9:
            if o < lo or o > hi:
                return None
            continue
        t1, t2 = (lo - o) / d, (hi - o) / d
        if t1 > t2:
            t1, t2 = t2, t1
        tmin, tmax = max(tmin, t1), min(tmax, t2)
        if tmin > tmax:
            return None
    if tmax < 0:
        return None
    return max(tmin, 0.0)


def _ray_box_exit_t(origin, direction):
    """t at which the ray leaves the arena rectangle (ray starts inside it)."""
    tx = math.inf
    if direction[0] > 1e-9:
        tx = (ARENA_W - origin[0]) / direction[0]
    elif direction[0] < -1e-9:
        tx = (0 - origin[0]) / direction[0]
    ty = math.inf
    if direction[1] > 1e-9:
        ty = (ARENA_H - origin[1]) / direction[1]
    elif direction[1] < -1e-9:
        ty = (0 - origin[1]) / direction[1]
    return min(tx, ty)


# ------------------------------------------------------------------ game objects

@dataclass
class Ship:
    pos: np.ndarray
    vel: np.ndarray
    angle: float
    alive: bool = True
    mode: str = "ship"  # "ship" | "astronaut"
    respawn_timer: int = 0
    cooldown: int = 0
    weapon: str = "gun"
    laser_charges: int = 0
    ammo: int = MAX_AMMO
    reload_timer: int = 0
    bullets_in_flight: int = 0


@dataclass
class Asteroid:
    pos: np.ndarray
    vel: np.ndarray
    radius: float
    hp: int


@dataclass
class Bullet:
    pos: np.ndarray
    vel: np.ndarray
    owner: str


@dataclass
class Pickup:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    active: bool = False
    ttl: int = 0


@dataclass
class Effect:
    """Visual-only explosion/laser flash, drawn by envs/render.py."""
    kind: str  # "explosion" | "laser"
    pos: np.ndarray = None
    end: np.ndarray = None
    radius: float = 0.0
    timer: int = 0


def decode_action(flat_action) -> dict:
    """Box(3,) [turn_rate, thrust, shoot_signal] -> the game's native per-ship action."""
    move = np.clip(np.asarray(flat_action[:2], dtype=np.float32), -1.0, 1.0)
    shoot = int(flat_action[2] > 0.0)
    return {"move": move, "shoot": shoot}


def observation_space() -> spaces.Box:
    own_low = np.array([-1.2, -1.2, -1.6, -1.6, -1, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    own_high = np.array([1.2, 1.2, 1.6, 1.6, 1, 1, 1.6, 1.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    rel_low = np.array([-2.4, -2.4, -3.2, -3.2, -1, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    rel_high = np.array([2.4, 2.4, 3.2, 3.2, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    ast_low = np.tile(np.array([-2.4, -2.4, 0.0], dtype=np.float32), NUM_ASTEROIDS)
    ast_high = np.tile(np.array([2.4, 2.4, 0.1], dtype=np.float32), NUM_ASTEROIDS)
    pu_low = np.tile(np.array([-2.4, -2.4, 0.0], dtype=np.float32), PICKUP_SLOTS)
    pu_high = np.tile(np.array([2.4, 2.4, 1.0], dtype=np.float32), PICKUP_SLOTS)
    bullet_low = np.tile(np.array([-2.4, -2.4, -1.0, -1.0, 0.0], dtype=np.float32), 2 * BULLET_SLOTS)
    bullet_high = np.tile(np.array([2.4, 2.4, 1.0, 1.0, 1.0], dtype=np.float32), 2 * BULLET_SLOTS)
    low = np.concatenate([own_low, rel_low, ast_low, pu_low, bullet_low])
    high = np.concatenate([own_high, rel_high, ast_high, pu_high, bullet_high])
    return spaces.Box(low, high, dtype=np.float32)


# ------------------------------------------------------------------ the game

class AstroPartyGame:
    """Full two-ship game state. Public state you may read (e.g. for reward
    shaping): ships ({"ship1": Ship, "ship2": Ship}), asteroids, bullets,
    pickups, t (step counter)."""

    def __init__(self, random_spawn: bool = True):
        # random_spawn=False uses the old fixed spawn points (left/right of the planet).
        # Grading (envs/eval_astro_party_env.py) always uses random spawns.
        self.random_spawn = random_spawn
        self._np_random = None

    def reset(self, seed=None):
        if seed is not None or self._np_random is None:
            self._np_random = np.random.default_rng(seed)
        rng = self._np_random
        self.t = 0

        if self.random_spawn:
            p1, a1 = self._random_spawn()
            p2, a2 = np.array([ARENA_W - p1[0], ARENA_H - p1[1]], dtype=np.float32), a1 + math.pi
        else:
            jitter = lambda: rng.uniform(-0.25, 0.25)
            p1, a1 = np.array([ARENA_W * 0.2, ARENA_H / 2], dtype=np.float32), math.pi / 2 + jitter()
            p2, a2 = np.array([ARENA_W * 0.8, ARENA_H / 2], dtype=np.float32), -math.pi / 2 + jitter()
        self.ships = {
            "ship1": Ship(pos=p1, vel=np.zeros(2, np.float32), angle=a1),
            "ship2": Ship(pos=p2, vel=np.zeros(2, np.float32), angle=a2),
        }
        avoid = [s.pos for s in self.ships.values()]
        self.asteroids = [self._spawn_asteroid(avoid) for _ in range(NUM_ASTEROIDS)]
        self.bullets: list[Bullet] = []
        self.pickups = [Pickup() for _ in range(PICKUP_SLOTS)]
        self.effects: list[Effect] = []
        # Background stars are only drawn, but they come from the seeded RNG, so
        # generating them here keeps every later random draw (asteroid respawns,
        # pickup drops) identical for a given seed.
        self.stars = [
            (rng.uniform(0, ARENA_W), rng.uniform(0, ARENA_H), rng.uniform(1, 2.4), rng.integers(120, 255))
            for _ in range(140)
        ]
        # One zero-action warm-up tick, so the first observation is post-physics.
        zero_action = {"move": np.zeros(2, np.float32), "shoot": 0}
        self.step({"ship1": zero_action, "ship2": zero_action})

    @property
    def done(self) -> bool:
        """True once either player has been eliminated."""
        return not (self.ships["ship1"].alive and self.ships["ship2"].alive)

    @property
    def truncated(self) -> bool:
        """True once the step limit is reached (a draw if nobody was eliminated)."""
        return self.t >= MAX_STEPS

    # ---------------------------------------------------------------- spawning

    def _rand_pos(self, radius, avoid, min_sep):
        rng = self._np_random
        wall_clear = math.hypot(BRACKET_W / 2, BRACKET_H / 2) + radius + 10
        for _ in range(50):
            pos = np.array([rng.uniform(radius, ARENA_W - radius), rng.uniform(radius, ARENA_H - radius)], dtype=np.float32)
            if np.hypot(*(pos - PLANET_POS)) < PLANET_RADIUS + radius + 10:
                continue
            if any(np.hypot(*(pos - a)) < min_sep for a in avoid):
                continue
            if any(math.hypot(pos[0] - cx, pos[1] - cy) < wall_clear for cx, cy in BRACKET_CENTERS):
                continue
            return pos
        return pos

    def _random_spawn(self):
        """A safe spot for ship1 whose mirror image (ship2's spot) is >= MIN_SPAWN_DIST away."""
        for _ in range(100):
            pos = self._rand_pos(SHIP_RADIUS + SPAWN_CLEARANCE, [], 0.0)
            mirror = np.array([ARENA_W - pos[0], ARENA_H - pos[1]], dtype=np.float32)
            if np.hypot(*(pos - mirror)) >= MIN_SPAWN_DIST:
                break
        angle = float(self._np_random.uniform(-math.pi, math.pi))
        return pos.astype(np.float32), angle

    def _spawn_asteroid(self, avoid):
        rng = self._np_random
        radius = float(rng.uniform(*ASTEROID_RADIUS_RANGE))
        pos = self._rand_pos(radius, avoid, radius + 60)
        speed = float(rng.uniform(*ASTEROID_SPEED_RANGE))
        theta = float(rng.uniform(0, 2 * math.pi))
        vel = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32) * speed
        return Asteroid(pos=pos, vel=vel, radius=radius, hp=ASTEROID_HP)

    def _try_spawn_pickup(self, pos):
        rng = self._np_random
        if rng.uniform() > PICKUP_DROP_PROB:
            return
        for slot in self.pickups:
            if not slot.active:
                slot.pos = pos.copy()
                slot.active = True
                slot.ttl = PICKUP_TTL_STEPS
                return

    def _destroy_asteroid(self, asteroid, avoid):
        self.effects.append(Effect(kind="explosion", pos=asteroid.pos.copy(), radius=asteroid.radius, timer=15))
        self._try_spawn_pickup(asteroid.pos)
        new = self._spawn_asteroid(avoid)
        asteroid.pos, asteroid.vel, asteroid.radius, asteroid.hp = new.pos, new.vel, new.radius, new.hp

    # ---------------------------------------------------------------- pilots

    def _eject_pilot(self, ship):
        """Ship destroyed: pop the pilot out as a small, gun-less, slower astronaut."""
        ship.mode = "astronaut"
        ship.respawn_timer = RESPAWN_STEPS
        ship.weapon = "gun"
        ship.laser_charges = 0
        ship.cooldown = 0
        ship.vel *= 0.4
        self.effects.append(Effect(kind="explosion", pos=ship.pos.copy(), radius=SHIP_RADIUS * 1.6, timer=18))

    def _eliminate(self, ship):
        """Astronaut killed: final elimination for that player."""
        ship.alive = False
        self.effects.append(Effect(kind="explosion", pos=ship.pos.copy(), radius=ASTRONAUT_RADIUS * 2.2, timer=15))

    def _respawn_pilot(self, ship):
        ship.mode = "ship"
        ship.vel = np.zeros(2, dtype=np.float32)
        ship.cooldown = 0
        ship.weapon = "gun"
        ship.laser_charges = 0
        ship.ammo = MAX_AMMO
        ship.reload_timer = 0

    def _hit(self, shooter_id, target_id, events):
        """A bullet/laser from shooter_id hit target_id: eject a ship, kill an astronaut."""
        target = self.ships[target_id]
        if target.mode == "ship":
            self._eject_pilot(target)
            events.append(Event("eject", by=shooter_id, target=target_id))
        else:
            self._eliminate(target)
            events.append(Event("kill", by=shooter_id, target=target_id))

    # ---------------------------------------------------------------- weapons

    def _fire_gun(self, ship_id, ship):
        if ship.ammo <= 0 or ship.bullets_in_flight >= BULLET_SLOTS:
            return
        nose = ship.pos + _rotate(SHIP_POLY_LOCAL[0:1], ship.angle)[0]
        vel = _heading(ship.angle) * BULLET_SPEED
        self.bullets.append(Bullet(pos=nose, vel=vel, owner=ship_id))
        ship.ammo -= 1
        ship.bullets_in_flight += 1
        if ship.reload_timer <= 0:
            ship.reload_timer = RELOAD_STEPS_PER_AMMO
        ship.cooldown = FIRE_COOLDOWN_STEPS

    def _fire_laser(self, ship_id, ship, events):
        opponent_id = other(ship_id)
        opponent = self.ships[opponent_id]
        origin = ship.pos + _rotate(SHIP_POLY_LOCAL[0:1], ship.angle)[0]
        direction = _heading(ship.angle)
        best_t, kind = _ray_box_exit_t(origin, direction), None

        t = _ray_circle_t(origin, direction, PLANET_POS, PLANET_RADIUS)
        if t is not None and t < best_t:
            best_t, kind = t, "planet"
        if opponent.alive:
            t = _ray_circle_t(origin, direction, opponent.pos, _radius(opponent))
            if t is not None and t < best_t:
                best_t, kind = t, "ship"
        hit_asteroid = None
        for a in self.asteroids:
            t = _ray_circle_t(origin, direction, a.pos, a.radius)
            if t is not None and t < best_t:
                best_t, kind, hit_asteroid = t, "asteroid", a
        for cx, cy, hw, hh in WALLS:
            t = _ray_aabb_entry_t(origin, direction, cx, cy, hw, hh)
            if t is not None and t < best_t:
                best_t, kind, hit_asteroid = t, "wall", None

        self.effects.append(Effect(kind="laser", pos=origin, end=origin + direction * best_t, timer=8))
        ship.cooldown = FIRE_COOLDOWN_STEPS
        ship.laser_charges -= 1
        if ship.laser_charges <= 0:
            ship.weapon = "gun"

        if kind == "ship":
            self._hit(ship_id, opponent_id, events)
        elif kind == "asteroid":
            events.append(Event("asteroid_hit", by=ship_id))
            self._destroy_asteroid(hit_asteroid, [s.pos for s in self.ships.values()])

    # ---------------------------------------------------------------- step

    def step(self, actions) -> list[Event]:
        """Advance the game one tick. `actions` maps ship id -> native action
        ({"move": [turn, thrust], "shoot": 0/1}). Returns this tick's events."""
        events: list[Event] = []

        for ship_id, ship in self.ships.items():
            if not ship.alive:
                continue
            act = actions[ship_id]
            move = np.clip(np.asarray(act["move"], dtype=np.float32), -1, 1)

            piloting = ship.mode == "ship"
            max_turn = MAX_TURN_RATE if piloting else ASTRONAUT_TURN_RATE
            accel_mag = THRUST_ACCEL if piloting else ASTRONAUT_ACCEL
            max_speed = MAX_SPEED if piloting else ASTRONAUT_MAX_SPEED
            drag = LINEAR_DRAG if piloting else ASTRONAUT_DRAG

            turn_rate, thrust = float(move[0]), float(move[1])
            ship.angle += turn_rate * max_turn * DT
            accel = accel_mag * (thrust if thrust >= 0 else thrust * REVERSE_ACCEL_SCALE)
            ship.vel += _heading(ship.angle) * accel * DT
            speed = float(np.hypot(*ship.vel))
            if speed > max_speed:
                ship.vel *= max_speed / speed
            ship.vel *= drag

            if piloting:
                if ship.cooldown > 0:
                    ship.cooldown -= 1
                if ship.ammo < MAX_AMMO:
                    ship.reload_timer -= 1
                    if ship.reload_timer <= 0:
                        ship.ammo += 1
                        ship.reload_timer = RELOAD_STEPS_PER_AMMO if ship.ammo < MAX_AMMO else 0
                if int(act["shoot"]) == 1 and ship.cooldown <= 0:
                    if ship.weapon == "laser" and ship.laser_charges > 0:
                        self._fire_laser(ship_id, ship, events)
                    else:
                        self._fire_gun(ship_id, ship)
            else:
                ship.respawn_timer -= 1
                if ship.respawn_timer <= 0:
                    self._respawn_pilot(ship)
                    events.append(Event("respawn", by=ship_id))

        for ship in self.ships.values():
            if ship.alive:
                ship.pos += ship.vel * DT
        for a in self.asteroids:
            a.pos += a.vel * DT

        alive_ships = [s for s in self.ships.values() if s.alive]
        for ship in alive_ships:
            r = _radius(ship)
            _wall_bounce(ship.pos, ship.vel, r)
            _resolve_circle_collision(ship.pos, ship.vel, 1.0, r, PLANET_POS, np.zeros(2, np.float32), 0.0, PLANET_RADIUS, PLANET_RESTITUTION)
            for cx, cy, hw, hh in WALLS:
                _circle_rect_collide(ship.pos, ship.vel, r, cx, cy, hw, hh, WALL_RESTITUTION)
        for a in self.asteroids:
            _wall_bounce(a.pos, a.vel, a.radius)
            mass = (a.radius / 20.0) ** 2
            _resolve_circle_collision(a.pos, a.vel, 1.0 / mass, a.radius, PLANET_POS, np.zeros(2, np.float32), 0.0, PLANET_RADIUS, PLANET_RESTITUTION)
            for cx, cy, hw, hh in WALLS:
                _circle_rect_collide(a.pos, a.vel, a.radius, cx, cy, hw, hh, WALL_RESTITUTION)

        # Ship vs ship: two ships bounce; a ship that touches an astronaut kills it.
        s1, s2 = self.ships["ship1"], self.ships["ship2"]
        if s1.alive and s2.alive:
            r1, r2 = _radius(s1), _radius(s2)
            if float(np.hypot(*(s1.pos - s2.pos))) < r1 + r2:
                if s1.mode == "ship" and s2.mode == "ship":
                    _resolve_circle_collision(s1.pos, s1.vel, 1.0, r1, s2.pos, s2.vel, 1.0, r2, BODY_RESTITUTION)
                elif s1.mode == "ship" and s2.mode == "astronaut":
                    self._eliminate(s2)
                    events.append(Event("kill", by="ship1", target="ship2"))
                elif s2.mode == "ship" and s1.mode == "astronaut":
                    self._eliminate(s1)
                    events.append(Event("kill", by="ship2", target="ship1"))

        for ship in alive_ships:
            r = _radius(ship)
            for a in self.asteroids:
                mass = (a.radius / 20.0) ** 2
                _resolve_circle_collision(ship.pos, ship.vel, 1.0, r, a.pos, a.vel, 1.0 / mass, a.radius, BODY_RESTITUTION)
        for i in range(len(self.asteroids)):
            for j in range(i + 1, len(self.asteroids)):
                a, b = self.asteroids[i], self.asteroids[j]
                ma, mb = (a.radius / 20.0) ** 2, (b.radius / 20.0) ** 2
                _resolve_circle_collision(a.pos, a.vel, 1.0 / ma, a.radius, b.pos, b.vel, 1.0 / mb, b.radius, BODY_RESTITUTION)

        surviving_bullets = []
        for bullet in self.bullets:
            removed = False

            step_vec = bullet.vel * DT
            step_dist = float(np.hypot(*step_vec))
            if step_dist > 1e-9:
                direction = step_vec / step_dist
                for cx, cy, hw, hh in WALLS:
                    t = _ray_aabb_entry_t(bullet.pos, direction, cx, cy, hw, hh)
                    if t is not None and t <= step_dist:
                        bullet.pos = bullet.pos + direction * t
                        removed = True
                        break
            if not removed:
                bullet.pos = bullet.pos + step_vec

            if removed:
                pass
            elif bullet.pos[0] < 0 or bullet.pos[0] > ARENA_W or bullet.pos[1] < 0 or bullet.pos[1] > ARENA_H:
                removed = True
            elif np.hypot(*(bullet.pos - PLANET_POS)) < PLANET_RADIUS + BULLET_RADIUS:
                removed = True
            else:
                for opp_id, opp in self.ships.items():
                    if opp_id != bullet.owner and opp.alive and np.hypot(*(bullet.pos - opp.pos)) < _radius(opp) + BULLET_RADIUS:
                        self._hit(bullet.owner, opp_id, events)
                        removed = True
                        break
                if not removed:
                    for a in self.asteroids:
                        if np.hypot(*(bullet.pos - a.pos)) < a.radius + BULLET_RADIUS:
                            a.hp -= 1
                            events.append(Event("asteroid_hit", by=bullet.owner))
                            if a.hp <= 0:
                                self._destroy_asteroid(a, [s.pos for s in self.ships.values()])
                            removed = True
                            break
            if removed:
                self.ships[bullet.owner].bullets_in_flight -= 1
            else:
                surviving_bullets.append(bullet)
        self.bullets = surviving_bullets

        for slot in self.pickups:
            if not slot.active:
                continue
            slot.ttl -= 1
            if slot.ttl <= 0:
                slot.active = False
                continue
            for ship_id, ship in self.ships.items():
                if ship.alive and ship.mode == "ship" and np.hypot(*(slot.pos - ship.pos)) < SHIP_RADIUS + PICKUP_RADIUS:
                    ship.weapon = "laser"
                    ship.laser_charges = 1
                    events.append(Event("pickup", by=ship_id))
                    slot.active = False
                    break

        for e in self.effects:
            e.timer -= 1
        self.effects = [e for e in self.effects if e.timer > 0]

        self.t += 1
        return events

    # ---------------------------------------------------------------- observation

    def observe(self, ship_id):
        """The Box(84,) observation from ship_id's point of view (layout in the module docstring)."""
        me = self.ships[ship_id]
        opp_id = other(ship_id)
        opp = self.ships[opp_id]

        hw, hh = ARENA_W / 2, ARENA_H / 2
        dist_planet = float(np.hypot(*(me.pos - PLANET_POS))) / hw
        dist_wall = min(
            math.hypot(
                max(cx - w2, min(me.pos[0], cx + w2)) - me.pos[0],
                max(cy - h2, min(me.pos[1], cy + h2)) - me.pos[1],
            )
            for cx, cy, w2, h2 in WALLS
        ) / hw

        own = [
            (me.pos[0] - hw) / hw,
            (me.pos[1] - hh) / hh,
            me.vel[0] / MAX_SPEED,
            me.vel[1] / MAX_SPEED,
            math.cos(me.angle),
            math.sin(me.angle),
            dist_planet,
            dist_wall,
            me.ammo / MAX_AMMO,
            1.0 if me.ammo >= MAX_AMMO else 1.0 - me.reload_timer / RELOAD_STEPS_PER_AMMO,
            1.0 if me.weapon == "laser" and me.laser_charges > 0 else 0.0,
            1.0 if me.cooldown <= 0 else 0.0,
            1.0 if me.mode == "astronaut" else 0.0,
            (me.respawn_timer / RESPAWN_STEPS) if me.mode == "astronaut" else 0.0,
        ]

        rel_angle = opp.angle - me.angle
        rel = [
            (opp.pos[0] - me.pos[0]) / hw,
            (opp.pos[1] - me.pos[1]) / hh,
            (opp.vel[0] - me.vel[0]) / MAX_SPEED,
            (opp.vel[1] - me.vel[1]) / MAX_SPEED,
            math.cos(rel_angle),
            math.sin(rel_angle),
            1.0 if opp.alive else 0.0,
            1.0 if opp.mode == "astronaut" else 0.0,
            opp.ammo / MAX_AMMO,
            1.0 if opp.ammo >= MAX_AMMO else 1.0 - opp.reload_timer / RELOAD_STEPS_PER_AMMO,
            1.0 if opp.weapon == "laser" and opp.laser_charges > 0 else 0.0,
            1.0 if opp.cooldown <= 0 else 0.0,
        ]

        ast = []
        for a in self.asteroids:
            ast += [(a.pos[0] - me.pos[0]) / hw, (a.pos[1] - me.pos[1]) / hh, a.radius / hw]

        pu = []
        for slot in self.pickups:
            if slot.active:
                pu += [(slot.pos[0] - me.pos[0]) / hw, (slot.pos[1] - me.pos[1]) / hh, 1.0]
            else:
                pu += [0.0, 0.0, 0.0]

        def bullet_slots(owner_id):
            live = [b for b in self.bullets if b.owner == owner_id]
            live.sort(key=lambda b: float(np.hypot(*(b.pos - me.pos))))
            out = []
            for b in live[:BULLET_SLOTS]:
                out += [
                    (b.pos[0] - me.pos[0]) / hw,
                    (b.pos[1] - me.pos[1]) / hh,
                    b.vel[0] / BULLET_SPEED,
                    b.vel[1] / BULLET_SPEED,
                    1.0,
                ]
            out += [0.0, 0.0, 0.0, 0.0, 0.0] * (BULLET_SLOTS - len(live[:BULLET_SLOTS]))
            return out

        own_bullets = bullet_slots(ship_id)
        opp_bullets = bullet_slots(opp_id)

        return np.array(own + rel + ast + pu + own_bullets + opp_bullets, dtype=np.float32)
