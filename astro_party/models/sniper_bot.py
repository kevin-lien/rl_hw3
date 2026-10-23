"""Sniper: the rule-based TA bot used as the strong baseline in eval_astro_party.py.
The same RuleBot class, with different style parameters, is the medium baseline
(models/rusher_bot.py).

Same obs -> action interface as every other opponent (a trained model wrapped as
`lambda obs: model.predict(obs, deterministic=True)[0]` works the same way):
takes one ship's Box(84,) observation and returns a flat Box(3,) action
[turn_rate, thrust, shoot_signal], shoot_signal > 0 meaning "fire".

What it does, in order of priority:
  - reads the observation back into world units (the env normalizes x/dx by 600
    and y/dy by 400, so the raw normalized vector is not a direction),
  - while it is a ship: keeps its preferred stand-off distance, leads its shots at the
    opponent's predicted intercept point, and only fires with a clear line of
    sight (planet, walls, asteroids); if the opponent hides behind the planet
    it orbits the planet to flank it,
  - when the opponent is an ejected pilot: chases it down (shoot or ram),
  - when it is the ejected pilot itself: hides behind the planet,
  - if dodging is on: dodges incoming bullets with a small model-predictive search
    over its own ship kinematics.
It keeps no state between calls, so it is fully deterministic.
"""
import math

import numpy as np

from envs.game import (
    ASTRONAUT_ACCEL, ASTRONAUT_DRAG, ASTRONAUT_MAX_SPEED, ASTRONAUT_RADIUS,
    BULLET_RADIUS, BULLET_SPEED, BULLET_TTL_STEPS, DT, LINEAR_DRAG, MAX_SPEED,
    MAX_TURN_RATE, PLANET_POS, PLANET_RADIUS, SHIP_RADIUS, THRUST_ACCEL, WALLS,
)

HW, HH = 600.0, 400.0  # observation normalization: x / 600, y / 400
TURN_STEP = MAX_TURN_RATE * DT
PX, PY = float(PLANET_POS[0]), float(PLANET_POS[1])
BULLET_RANGE_T = BULLET_TTL_STEPS * DT


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _seg_circle_blocked(ax, ay, bx, by, cx, cy, r):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return math.hypot(ax - cx, ay - cy) < r
    t = ((cx - ax) * dx + (cy - ay) * dy) / L2
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    return math.hypot(ax + t * dx - cx, ay + t * dy - cy) < r


def _seg_rect_blocked(ax, ay, bx, by, cx, cy, hw, hh):
    t0, t1 = 0.0, 1.0
    for o, d, lo, hi in ((ax, bx - ax, cx - hw, cx + hw), (ay, by - ay, cy - hh, cy + hh)):
        if abs(d) < 1e-9:
            if o < lo or o > hi:
                return False
            continue
        u0, u1 = (lo - o) / d, (hi - o) / d
        if u0 > u1:
            u0, u1 = u1, u0
        t0, t1 = max(t0, u0), min(t1, u1)
        if t0 > t1:
            return False
    return True


def line_of_sight(ax, ay, bx, by, asteroids=()):
    if _seg_circle_blocked(ax, ay, bx, by, PX, PY, PLANET_RADIUS + BULLET_RADIUS):
        return False
    for cx, cy, hw, hh in WALLS:
        if _seg_rect_blocked(ax, ay, bx, by, cx, cy, hw + BULLET_RADIUS, hh + BULLET_RADIUS):
            return False
    for cx, cy, r in asteroids:
        if _seg_circle_blocked(ax, ay, bx, by, cx, cy, r + BULLET_RADIUS):
            return False
    return True


def intercept_time(dx, dy, vx, vy, speed):
    """Smallest t>0 with |d + v t| = speed * t (target moving at v, projectile at speed)."""
    a = vx * vx + vy * vy - speed * speed
    b = 2 * (dx * vx + dy * vy)
    c = dx * dx + dy * dy
    if abs(a) < 1e-9:
        return c / max(-b, 1e-9) if b < 0 else None
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    sq = math.sqrt(disc)
    ts = [t for t in ((-b - sq) / (2 * a), (-b + sq) / (2 * a)) if t > 0]
    return min(ts) if ts else None


# Observation indices (layout documented in envs/game.py).
X, Y, VX, VY, COS, SIN = 0, 1, 2, 3, 4, 5
AMMO, LASER, READY, ASTRONAUT = 8, 10, 11, 12
OPP_DX, OPP_DY, OPP_DVX, OPP_DVY = 14, 15, 16, 17
OPP_ALIVE, OPP_ASTRONAUT = 20, 21
ASTEROIDS = 26      # 4 slots of (dx, dy, radius)
OPP_BULLETS = 64    # 4 slots of (dx, dy, vx, vy, active)


class State:
    """The observation decoded into world units."""
    __slots__ = ("px", "py", "vx", "vy", "ang", "hx", "hy", "dx", "dy", "ovx", "ovy",
                 "opp_alive", "opp_astro", "me_astro", "ammo", "ready", "laser",
                 "asteroids", "opp_bullets")

    def __init__(self, o):
        self.px, self.py = o[X] * HW + HW, o[Y] * HH + HH
        self.vx, self.vy = o[VX] * MAX_SPEED, o[VY] * MAX_SPEED
        self.ang = math.atan2(o[SIN], o[COS])
        self.hx, self.hy = math.sin(self.ang), -math.cos(self.ang)
        self.dx, self.dy = o[OPP_DX] * HW, o[OPP_DY] * HH
        self.ovx, self.ovy = self.vx + o[OPP_DVX] * MAX_SPEED, self.vy + o[OPP_DVY] * MAX_SPEED
        self.opp_alive = o[OPP_ALIVE] > 0.5
        self.opp_astro = o[OPP_ASTRONAUT] > 0.5
        self.me_astro = o[ASTRONAUT] > 0.5
        self.ammo = int(round(o[AMMO] * 3))
        self.ready = o[READY] > 0.5
        self.laser = o[LASER] > 0.5
        self.asteroids = []
        for k in range(4):
            a = ASTEROIDS + 3 * k
            self.asteroids.append((self.px + o[a] * HW, self.py + o[a + 1] * HH, o[a + 2] * HW))
        self.opp_bullets = []
        for k in range(4):
            b = OPP_BULLETS + 5 * k
            if o[b + 4] > 0.5:
                self.opp_bullets.append((self.px + o[b] * HW, self.py + o[b + 1] * HH,
                                         o[b + 2] * BULLET_SPEED, o[b + 3] * BULLET_SPEED))


def _sim_min_dist(s, turn, thrust, bullets, steps, astro):
    """Roll our own kinematics forward under a fixed (turn, thrust) and return
    the closest approach to any of the given bullets."""
    accel_mag = ASTRONAUT_ACCEL if astro else THRUST_ACCEL
    max_speed = ASTRONAUT_MAX_SPEED if astro else MAX_SPEED
    drag = ASTRONAUT_DRAG if astro else LINEAR_DRAG
    px, py, vx, vy, ang = s.px, s.py, s.vx, s.vy, s.ang
    best = 1e9
    acc = accel_mag * (thrust if thrust >= 0 else thrust * 0.5) * DT
    for k in range(1, steps + 1):
        ang += turn * TURN_STEP
        vx += math.sin(ang) * acc
        vy += -math.cos(ang) * acc
        sp = math.hypot(vx, vy)
        if sp > max_speed:
            vx *= max_speed / sp
            vy *= max_speed / sp
        vx *= drag
        vy *= drag
        px += vx * DT
        py += vy * DT
        for bx, by, bvx, bvy in bullets:
            d = math.hypot(px - (bx + bvx * DT * k), py - (by + bvy * DT * k))
            if d < best:
                best = d
    return best


def _threatening(s, bullets, radius):
    """Bullets whose straight-line path passes near our current drift path soon."""
    out = []
    for bx, by, bvx, bvy in bullets:
        rx, ry = bx - s.px, by - s.py
        rvx, rvy = bvx - s.vx, bvy - s.vy
        v2 = rvx * rvx + rvy * rvy
        if v2 < 1e-6:
            continue
        t = -(rx * rvx + ry * rvy) / v2
        if t < 0 or t > 0.6:
            continue
        if math.hypot(rx + rvx * t, ry + rvy * t) < radius + 30:
            out.append((bx, by, bvx, bvy))
    return out


def _dodge(s, pref_turn, pref_thrust, astro):
    radius = (ASTRONAUT_RADIUS if astro else SHIP_RADIUS) + BULLET_RADIUS
    threats = _threatening(s, s.opp_bullets, radius)
    if not threats:
        return pref_turn, pref_thrust
    margin = radius + 8
    base = _sim_min_dist(s, pref_turn, pref_thrust, threats, 18, astro)
    if base >= margin:
        return pref_turn, pref_thrust
    best, best_score = (pref_turn, pref_thrust), base
    for turn in (-1.0, 0.0, 1.0):
        for thrust in (1.0, 0.0, -1.0):
            d = _sim_min_dist(s, turn, thrust, threats, 18, astro)
            score = min(d, margin + 5) - 0.5 * abs(turn - pref_turn)
            if score > best_score + 1e-6:
                best, best_score = (turn, thrust), score
    return best


def _flank_target(s):
    """Opponent hidden behind the planet: orbit the planet toward it."""
    sx, sy = s.px - PX, s.py - PY
    ox, oy = s.px + s.dx - PX, s.py + s.dy - PY
    th_s, th_o = math.atan2(sy, sx), math.atan2(oy, ox)
    direction = 1.0 if _wrap(th_o - th_s) > 0 else -1.0
    r = min(max(math.hypot(sx, sy), PLANET_RADIUS + 110), 330.0)
    th = th_s + direction * 0.7
    return PX + r * math.cos(th) - s.px, PY + r * math.sin(th) - s.py


def _flee(s):
    ox, oy = s.px + s.dx, s.py + s.dy
    # hide point: far side of the planet from the opponent
    ux, uy = PX - ox, PY - oy
    n = math.hypot(ux, uy) or 1.0
    hx, hy = PX + ux / n * (PLANET_RADIUS + 45), PY + uy / n * (PLANET_RADIUS + 45)
    if line_of_sight(ox, oy, s.px, s.py, ()):
        tx, ty = hx - s.px, hy - s.py
    else:
        tx, ty = -s.dx, -s.dy  # already hidden: keep distance
    err = _wrap(math.atan2(tx, -ty) - s.ang)
    turn = max(-1.0, min(1.0, err / TURN_STEP))
    thrust = 1.0 if abs(err) < 0.9 else 0.3
    return turn, thrust, False


class RuleBot:
    """Rule-based duelist; the style parameters decide how it fights.
      near, far      keeps the opponent between these distances (px): backs off
                     inside `near`, closes in beyond `far`
      fire_tol       aim tolerance, as a fraction of the target's apparent size
                     (lower = pickier, more accurate shots)
      dodge          whether it dodges incoming bullets
      max_shot_time  never fires a bullet that would need longer than this (s) to arrive
    """

    def __init__(self, near, far, fire_tol, dodge=True, max_shot_time=1.1):
        self.near, self.far = near, far
        self.fire_tol = fire_tol
        self.dodge = dodge
        self.max_shot_time = max_shot_time

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        s = State(obs)
        if not s.opp_alive:
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)
        if s.me_astro:
            turn, thrust, shoot = _flee(s)
        elif s.opp_astro:
            turn, thrust, shoot = self._hunt(s)
        else:
            turn, thrust, shoot = self._duel(s)
        if self.dodge:
            turn, thrust = _dodge(s, turn, thrust, s.me_astro)
        return np.array([turn, thrust, 1.0 if shoot else -1.0], dtype=np.float32)

    def _aim(self, s, target_r):
        dx, dy = s.dx, s.dy
        if s.laser:
            ax, ay, t = dx, dy, math.hypot(dx, dy) / BULLET_SPEED
        else:
            t = intercept_time(dx, dy, s.ovx, s.ovy, BULLET_SPEED)
            if t is None:
                t = math.hypot(dx, dy) / BULLET_SPEED
            ax, ay = dx + s.ovx * t, dy + s.ovy * t
        err = _wrap(math.atan2(ax, -ay) - s.ang)  # heading(angle) = (sin, -cos)
        dist = math.hypot(ax, ay)
        tol = math.atan2(target_r * 0.8, max(dist, 1.0)) * self.fire_tol
        clear = line_of_sight(s.px + 20 * s.hx, s.py + 20 * s.hy, s.px + ax, s.py + ay, s.asteroids)
        can_fire = s.ready and (s.laser or s.ammo > 0) and clear and abs(err) < tol
        if not s.laser and t > min(self.max_shot_time, BULLET_RANGE_T * 0.95):
            can_fire = False
        return err, can_fire

    def _duel(self, s):
        err, can_fire = self._aim(s, SHIP_RADIUS + BULLET_RADIUS)
        turn = max(-1.0, min(1.0, err / TURN_STEP))
        dist = math.hypot(s.dx, s.dy)
        speed = math.hypot(s.vx, s.vy)
        if not line_of_sight(s.px, s.py, s.px + s.dx, s.py + s.dy, ()):
            tx, ty = _flank_target(s)
            nerr = _wrap(math.atan2(tx, -ty) - s.ang)
            turn = max(-1.0, min(1.0, nerr / TURN_STEP))
            thrust = 1.0 if abs(nerr) < 0.6 else (0.3 if speed < 120 else 0.0)
            return turn, thrust, False
        if dist > self.far:
            thrust = 1.0 if abs(err) < 0.6 else 0.2
        elif dist < self.near:
            thrust = -1.0 if abs(err) < 1.2 else 0.0
        else:
            thrust = -0.6 if speed > 140 else 0.15
        return turn, thrust, can_fire

    def _hunt(self, s):
        _, can_fire = self._aim(s, ASTRONAUT_RADIUS + BULLET_RADIUS)
        dist = math.hypot(s.dx, s.dy)
        t = dist / 220.0
        tx, ty = s.dx + s.ovx * t, s.dy + s.ovy * t
        if not line_of_sight(s.px, s.py, s.px + tx, s.py + ty, ()):
            tx, ty = _flank_target(s)
        err = _wrap(math.atan2(tx, -ty) - s.ang)
        turn = max(-1.0, min(1.0, err / TURN_STEP))
        thrust = 1.0 if abs(err) < 0.7 else 0.25
        return turn, thrust, can_fire


# Sniper: long stand-off distance, picky aim, dodges.
sniper_bot_action = RuleBot(near=380.0, far=620.0, fire_tol=0.8)
