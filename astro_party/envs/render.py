"""Pygame drawing for AstroPartyGame. Purely visual: nothing here affects the game.
DON'T MODIFY: the TAs grade with the original version of this file.
"""
import math

import numpy as np
from gymnasium.error import DependencyNotInstalled

from envs.game import (
    ARENA_H, ARENA_W, ASTRONAUT_POLY_LOCAL, ASTRONAUT_RADIUS, BULLET_RADIUS, FPS,
    PICKUP_RADIUS, PLANET_POS, PLANET_RADIUS, RESPAWN_STEPS, SHIP_POLY_LOCAL, WALLS, _rotate,
)

SHIP_COLORS = {"ship1": ((80, 150, 235), (30, 70, 140)), "ship2": ((230, 90, 90), (140, 40, 40))}


class Renderer:
    """render_mode "human" opens a window; "rgb_array" returns an (H, W, 3) frame."""

    def __init__(self, render_mode):
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

    def draw(self, game):
        if self.render_mode is None:
            return None
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled('pygame is not installed, run `pip install pygame`') from e

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((int(ARENA_W), int(ARENA_H)))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.Surface((int(ARENA_W), int(ARENA_H)))
        surf.fill((6, 6, 14))
        for x, y, size, brightness in game.stars:
            gfxdraw.filled_circle(surf, int(x), int(y), max(1, int(size)), (brightness, brightness, brightness))

        for cx, cy, hw, hh in WALLS:
            rect = pygame.Rect(int(cx - hw), int(cy - hh), int(2 * hw), int(2 * hh))
            pygame.draw.rect(surf, (70, 55, 45), rect)
            pygame.draw.rect(surf, (215, 100, 30), rect, width=2)

        gfxdraw.filled_circle(surf, int(PLANET_POS[0]), int(PLANET_POS[1]), int(PLANET_RADIUS), (70, 70, 78))
        gfxdraw.aacircle(surf, int(PLANET_POS[0]), int(PLANET_POS[1]), int(PLANET_RADIUS), (110, 110, 120))
        rng = np.random.default_rng(0)  # fixed crater pattern
        for _ in range(10):
            r = rng.uniform(0.2, 0.85) * PLANET_RADIUS
            theta = rng.uniform(0, 2 * math.pi)
            cx = PLANET_POS[0] + math.cos(theta) * r
            cy = PLANET_POS[1] + math.sin(theta) * r
            gfxdraw.filled_circle(surf, int(cx), int(cy), int(rng.uniform(4, 12)), (50, 50, 58))

        for a in game.asteroids:
            gfxdraw.filled_circle(surf, int(a.pos[0]), int(a.pos[1]), int(a.radius), (150, 150, 150))
            gfxdraw.aacircle(surf, int(a.pos[0]), int(a.pos[1]), int(a.radius), (230, 230, 230))

        for slot in game.pickups:
            if slot.active:
                x, y = int(slot.pos[0]), int(slot.pos[1])
                pts = [(x, y - PICKUP_RADIUS), (x + PICKUP_RADIUS, y), (x, y + PICKUP_RADIUS), (x - PICKUP_RADIUS, y)]
                gfxdraw.filled_polygon(surf, pts, (60, 220, 220))
                gfxdraw.aapolygon(surf, pts, (200, 255, 255))

        for b in game.bullets:
            gfxdraw.filled_circle(surf, int(b.pos[0]), int(b.pos[1]), int(BULLET_RADIUS), (255, 230, 90))

        for e in game.effects:
            if e.kind == "explosion":
                progress = 1.0 - e.timer / 20.0
                r = int(e.radius * (0.6 + 1.4 * progress))
                alpha_color = (255, int(160 * (1 - progress)) + 40, 40)
                gfxdraw.aacircle(surf, int(e.pos[0]), int(e.pos[1]), max(1, r), alpha_color)
            elif e.kind == "laser":
                start, end = (int(e.pos[0]), int(e.pos[1])), (int(e.end[0]), int(e.end[1]))
                pygame.draw.line(surf, (40, 90, 235), start, end, 7)
                pygame.draw.line(surf, (170, 210, 255), start, end, 2)

        for ship_id, ship in game.ships.items():
            if not ship.alive:
                continue
            fill, outline = SHIP_COLORS[ship_id]
            if ship.mode == "ship":
                path = ship.pos + _rotate(SHIP_POLY_LOCAL, ship.angle)
                pts = [(int(x), int(y)) for x, y in path]
                pygame.draw.polygon(surf, fill, pts)
                gfxdraw.aapolygon(surf, pts, outline)
            else:
                path = ship.pos + _rotate(ASTRONAUT_POLY_LOCAL, ship.angle)
                pts = [(int(x), int(y)) for x, y in path]
                pygame.draw.polygon(surf, (222, 218, 205), pts)
                gfxdraw.aapolygon(surf, pts, outline)
                hx, hy = ship.pos + _rotate(ASTRONAUT_POLY_LOCAL[0:1], ship.angle)[0]
                gfxdraw.filled_circle(surf, int(hx), int(hy), 4, fill)
                gfxdraw.aacircle(surf, int(hx), int(hy), 4, outline)
                x, y, r = int(ship.pos[0]), int(ship.pos[1]), int(ASTRONAUT_RADIUS)
                progress = 1.0 - ship.respawn_timer / RESPAWN_STEPS
                ring = pygame.Rect(x - r - 4, y - r - 4, 2 * (r + 4), 2 * (r + 4))
                pygame.draw.arc(surf, fill, ring, -math.pi / 2, -math.pi / 2 + progress * 2 * math.pi, 3)

        if self.render_mode == "human":
            self.screen.blit(surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(FPS)
            pygame.display.flip()
            return None
        return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.screen = None
