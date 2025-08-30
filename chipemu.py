#!/usr/bin/env python3
"""
Enhanced Chip-8 / Super-CHIP Emulator in Python (single file)

Features added vs original version:
- Runtime compatibility flags:
  --legacy-store    : original FX55/FX65 increment I after store/load
  --legacy-shift    : original 8XY6/8XYE behavior (shift Vy into Vx before shift)
  --alt-bnnn        : alternative BNNN behavior (use Vx as offset instead of V0)
  --superchip       : enable Super-CHIP (SCHIP) features including 128x64 mode
- SCHIP features implemented:
  00FB - scroll right 4
  00FC - scroll left 4
  00CN - scroll down N
  00FE - switch to low-res (64x32)
  00FF - switch to high-res (128x64)
  DRW with n==0 draws a 16x16 sprite (2 bytes per row)
- Shift opcode behavior selectable via --legacy-shift
- Display supports both 64x32 and 128x64 with proper rendering and wrapping

Dependencies:
  - Python 3.8+
  - pygame (pip install pygame)

Run:
  python chip8_emulator.py path/to/rom [--scale 10] [--clock 700] [--superchip]

Author: D/V
License: MIT
"""

from __future__ import annotations
import argparse
import sys
import time
import random
from dataclasses import dataclass, field
from typing import List

try:
    import pygame
except Exception as e:
    print("This emulator requires pygame. Install with: pip install pygame", file=sys.stderr)
    raise

# ==============================
# Constants
# ==============================
MEM_SIZE = 4096
START_ADDRESS = 0x200
FONT_ADDRESS = 0x50  # canonical address for font sprites

# Default low-res Chip-8 screen
DEFAULT_W, DEFAULT_H = 64, 32

# Classic CHIP-8 4x5 font (each char 5 bytes)
FONTSET = [
    0xF0, 0x90, 0x90, 0x90, 0xF0,  # 0
    0x20, 0x60, 0x20, 0x20, 0x70,  # 1
    0xF0, 0x10, 0xF0, 0x80, 0xF0,  # 2
    0xF0, 0x10, 0xF0, 0x10, 0xF0,  # 3
    0x90, 0x90, 0xF0, 0x10, 0x10,  # 4
    0xF0, 0x80, 0xF0, 0x10, 0xF0,  # 5
    0xF0, 0x80, 0xF0, 0x90, 0xF0,  # 6
    0xF0, 0x10, 0x20, 0x40, 0x40,  # 7
    0xF0, 0x90, 0xF0, 0x90, 0xF0,  # 8
    0xF0, 0x90, 0xF0, 0x10, 0xF0,  # 9
    0xF0, 0x90, 0xF0, 0x90, 0x90,  # A
    0xE0, 0x90, 0xE0, 0x90, 0xE0,  # B
    0xF0, 0x80, 0x80, 0x80, 0xF0,  # C
    0xE0, 0x90, 0x90, 0x90, 0xE0,  # D
    0xF0, 0x80, 0xF0, 0x80, 0xF0,  # E
    0xF0, 0x80, 0xF0, 0x80, 0x80   # F
]

# Keyboard mapping: CHIP-8 key index -> pygame key
KEYMAP = {
    0x0: pygame.K_x,
    0x1: pygame.K_1,
    0x2: pygame.K_2,
    0x3: pygame.K_3,
    0x4: pygame.K_q,
    0x5: pygame.K_w,
    0x6: pygame.K_e,
    0x7: pygame.K_a,
    0x8: pygame.K_s,
    0x9: pygame.K_d,
    0xA: pygame.K_z,
    0xB: pygame.K_c,
    0xC: pygame.K_4,
    0xD: pygame.K_r,
    0xE: pygame.K_f,
    0xF: pygame.K_v,
}

@dataclass
class Chip8:
    legacy_store: bool = False
    legacy_shift: bool = False
    alt_bnnn: bool = False
    schip_mode: bool = False

    memory: bytearray = field(default_factory=lambda: bytearray(MEM_SIZE))
    V: List[int] = field(default_factory=lambda: [0] * 16)  # registers V0..VF
    I: int = 0
    pc: int = START_ADDRESS
    stack: List[int] = field(default_factory=list)
    delay_timer: int = 0
    sound_timer: int = 0
    width: int = DEFAULT_W
    height: int = DEFAULT_H
    display: List[int] = field(default_factory=lambda: [0] * (DEFAULT_W * DEFAULT_H))
    keys: List[bool] = field(default_factory=lambda: [False] * 16)
    draw_flag: bool = False

    def reset(self):
        self.memory = bytearray(MEM_SIZE)
        self.memory[FONT_ADDRESS:FONT_ADDRESS + len(FONTSET)] = bytes(FONTSET)
        self.V = [0] * 16
        self.I = 0
        self.pc = START_ADDRESS
        self.stack = []
        self.delay_timer = 0
        self.sound_timer = 0
        self.width = DEFAULT_W
        self.height = DEFAULT_H
        self.display = [0] * (self.width * self.height)
        self.keys = [False] * 16
        self.draw_flag = True

    def set_schip(self, enabled: bool):
        self.schip_mode = enabled
        if enabled:
            self.width = 128
            self.height = 64
        else:
            self.width = 64
            self.height = 32
        self.display = [0] * (self.width * self.height)
        self.draw_flag = True

    def load_rom(self, data: bytes):
        self.reset()
        end = START_ADDRESS + len(data)
        if end > MEM_SIZE:
            raise ValueError("ROM is too large for memory")
        self.memory[START_ADDRESS:end] = data

    # =============== Core fetch/decode/execute cycle ===============
    def fetch_opcode(self) -> int:
        hi = self.memory[self.pc]
        lo = self.memory[self.pc + 1]
        return (hi << 8) | lo

    def emulate_cycle(self):
        opcode = self.fetch_opcode()
        self.pc = (self.pc + 2) & 0xFFF

        nnn = opcode & 0x0FFF
        n = opcode & 0x000F
        x = (opcode & 0x0F00) >> 8
        y = (opcode & 0x00F0) >> 4
        kk = opcode & 0x00FF

        # 0x0000 group (including SCHIP scroll and mode ops)
        if opcode == 0x00E0:  # CLS
            self.display = [0] * (self.width * self.height)
            self.draw_flag = True
        elif opcode == 0x00EE:  # RET
            if not self.stack:
                raise RuntimeError("Stack underflow on RET")
            self.pc = self.stack.pop()
        elif opcode & 0xF000 == 0x0000:
            # SCHIP extended opcodes: 00CN, 00FB, 00FC, 00FE, 00FF
            if self.schip_mode and (opcode & 0xF0FF) == 0x00FB:
                # Scroll display right by 4 pixels
                self._scroll_right(4)
                self.draw_flag = True
            elif self.schip_mode and (opcode & 0xF0FF) == 0x00FC:
                # Scroll display left by 4 pixels
                self._scroll_left(4)
                self.draw_flag = True
            elif self.schip_mode and (opcode & 0xF0FF) == 0x00FE:
                # Low resolution
                self.set_schip(False)
            elif self.schip_mode and (opcode & 0xF0FF) == 0x00FF:
                # High resolution
                self.set_schip(True)
            elif self.schip_mode and (opcode & 0xF0F0) == 0x00C0:
                # 00CN - scroll down N lines (lowest nibble)
                n_down = opcode & 0x000F
                self._scroll_down(n_down)
                self.draw_flag = True
            else:
                # 0NNN - ignored or system call
                pass
        elif opcode & 0xF000 == 0x1000:  # JP addr
            self.pc = nnn
        elif opcode & 0xF000 == 0x2000:  # CALL addr
            self.stack.append(self.pc)
            self.pc = nnn
        elif opcode & 0xF000 == 0x3000:  # SE Vx, byte
            if self.V[x] == kk:
                self.pc += 2
        elif opcode & 0xF000 == 0x4000:  # SNE Vx, byte
            if self.V[x] != kk:
                self.pc += 2
        elif opcode & 0xF00F == 0x5000:  # SE Vx, Vy
            if self.V[x] == self.V[y]:
                self.pc += 2
        elif opcode & 0xF000 == 0x6000:  # LD Vx, byte
            self.V[x] = kk
        elif opcode & 0xF000 == 0x7000:  # ADD Vx, byte
            self.V[x] = (self.V[x] + kk) & 0xFF
        elif opcode & 0xF00F == 0x8000:  # LD Vx, Vy
            self.V[x] = self.V[y]
        elif opcode & 0xF00F == 0x8001:  # OR Vx, Vy
            self.V[x] |= self.V[y]
            self.V[x] &= 0xFF
        elif opcode & 0xF00F == 0x8002:  # AND Vx, Vy
            self.V[x] &= self.V[y]
        elif opcode & 0xF00F == 0x8003:  # XOR Vx, Vy
            self.V[x] ^= self.V[y]
        elif opcode & 0xF00F == 0x8004:  # ADD Vx, Vy
            total = self.V[x] + self.V[y]
            self.V[0xF] = 1 if total > 0xFF else 0
            self.V[x] = total & 0xFF
        elif opcode & 0xF00F == 0x8005:  # SUB Vx, Vy (Vx = Vx - Vy)
            self.V[0xF] = 1 if self.V[x] > self.V[y] else 0
            self.V[x] = (self.V[x] - self.V[y]) & 0xFF
        elif opcode & 0xF00F == 0x8006:  # SHR Vx {, Vy}
            if self.legacy_shift:
                # Original quirk: Vx = Vy; then shift Vx right
                self.V[x] = self.V[y]
            self.V[0xF] = self.V[x] & 0x1
            self.V[x] = (self.V[x] >> 1) & 0xFF
        elif opcode & 0xF00F == 0x8007:  # SUBN Vx, Vy (Vx = Vy - Vx)
            self.V[0xF] = 1 if self.V[y] > self.V[x] else 0
            self.V[x] = (self.V[y] - self.V[x]) & 0xFF
        elif opcode & 0xF00F == 0x800E:  # SHL Vx {, Vy}
            if self.legacy_shift:
                # Original quirk: Vx = Vy; then shift Vx left
                self.V[x] = self.V[y]
            self.V[0xF] = (self.V[x] >> 7) & 0x1
            self.V[x] = (self.V[x] << 1) & 0xFF
        elif opcode & 0xF00F == 0x9000:  # SNE Vx, Vy
            if self.V[x] != self.V[y]:
                self.pc += 2
        elif opcode & 0xF000 == 0xA000:  # LD I, addr
            self.I = nnn
        elif opcode & 0xF000 == 0xB000:  # JP V0, addr (or alt)
            if self.alt_bnnn:
                # alternative behavior: use Vx as offset
                self.pc = (nnn + self.V[x]) & 0xFFF
            else:
                self.pc = (nnn + self.V[0]) & 0xFFF
        elif opcode & 0xF000 == 0xC000:  # RND Vx, byte
            self.V[x] = random.randint(0, 255) & kk
        elif opcode & 0xF000 == 0xD000:  # DRW Vx, Vy, nibble
            # If SCHIP and n == 0 => 16x16 sprite (2 bytes per row)
            if self.schip_mode and n == 0:
                self._draw_sprite_16(self.V[x], self.V[y])
            else:
                self._draw_sprite_8(self.V[x], self.V[y], n)
        elif opcode & 0xF0FF == 0xE09E:  # SKP Vx
            if self._is_key_down(self.V[x] & 0xF):
                self.pc += 2
        elif opcode & 0xF0FF == 0xE0A1:  # SKNP Vx
            if not self._is_key_down(self.V[x] & 0xF):
                self.pc += 2
        elif opcode & 0xF0FF == 0xF007:  # LD Vx, DT
            self.V[x] = self.delay_timer
        elif opcode & 0xF0FF == 0xF00A:  # LD Vx, K (wait for key)
            key = self._wait_for_key()
            if key is None:
                # stall: back up PC so instruction repeats until key press
                self.pc = (self.pc - 2) & 0xFFF
            else:
                self.V[x] = key
        elif opcode & 0xF0FF == 0xF015:  # LD DT, Vx
            self.delay_timer = self.V[x]
        elif opcode & 0xF0FF == 0xF018:  # LD ST, Vx
            self.sound_timer = self.V[x]
        elif opcode & 0xF0FF == 0xF01E:  # ADD I, Vx
            self.I = (self.I + self.V[x]) & 0xFFFF
        elif opcode & 0xF0FF == 0xF029:  # LD F, Vx
            digit = self.V[x] & 0xF
            self.I = FONT_ADDRESS + digit * 5
        elif opcode & 0xF0FF == 0xF033:  # LD B, Vx (BCD)
            val = self.V[x]
            self.memory[self.I] = val // 100
            self.memory[self.I + 1] = (val // 10) % 10
            self.memory[self.I + 2] = val % 10
        elif opcode & 0xF0FF == 0xF055:  # LD [I], Vx
            for i in range(x + 1):
                self.memory[self.I + i] = self.V[i]
            if self.legacy_store:
                self.I = (self.I + x + 1) & 0xFFFF
        elif opcode & 0xF0FF == 0xF065:  # LD Vx, [I]
            for i in range(x + 1):
                self.V[i] = self.memory[self.I + i]
            if self.legacy_store:
                self.I = (self.I + x + 1) & 0xFFFF
        else:
            raise NotImplementedError(f"Unknown opcode: {opcode:04X} at PC {self.pc-2:03X}")

    # =============== Helpers ===============
    def _is_key_down(self, chip8_key: int) -> bool:
        if 0 <= chip8_key <= 0xF:
            return self.keys[chip8_key]
        return False

    def _wait_for_key(self) -> int | None:
        # Non-blocking check: if a key is pressed, return its CHIP-8 index; else None
        for i in range(16):
            if self.keys[i]:
                return i
        return None

    def _draw_sprite_8(self, x_pos: int, y_pos: int, height: int):
        self.V[0xF] = 0
        x_pos %= self.width
        y_pos %= self.height
        for row in range(height):
            sprite = self.memory[self.I + row]
            py = (y_pos + row) % self.height
            for col in range(8):
                bit = (sprite >> (7 - col)) & 1
                if bit:
                    px = (x_pos + col) % self.width
                    idx = py * self.width + px
                    if self.display[idx] == 1:
                        self.V[0xF] = 1
                    self.display[idx] ^= 1
        self.draw_flag = True

    def _draw_sprite_16(self, x_pos: int, y_pos: int):
        # 16x16 sprite: each row is two bytes (big-endian), total 32 bytes
        self.V[0xF] = 0
        x_pos %= self.width
        y_pos %= self.height
        for row in range(16):
            hi = self.memory[self.I + row*2]
            lo = self.memory[self.I + row*2 + 1]
            bits = (hi << 8) | lo
            py = (y_pos + row) % self.height
            for col in range(16):
                bit = (bits >> (15 - col)) & 1
                if bit:
                    px = (x_pos + col) % self.width
                    idx = py * self.width + px
                    if self.display[idx] == 1:
                        self.V[0xF] = 1
                    self.display[idx] ^= 1
        self.draw_flag = True

    def _scroll_right(self, pixels: int):
        w, h = self.width, self.height
        new = [0] * (w * h)
        for y in range(h):
            for x in range(w):
                src = y*w + x
                dst_x = (x + pixels) % w
                new[y*w + dst_x] = self.display[src]
        self.display = new

    def _scroll_left(self, pixels: int):
        w, h = self.width, self.height
        new = [0] * (w * h)
        for y in range(h):
            for x in range(w):
                src = y*w + x
                dst_x = (x - pixels) % w
                new[y*w + dst_x] = self.display[src]
        self.display = new

    def _scroll_down(self, lines: int):
        # scroll down by shifting rows downward; top rows become zero
        w, h = self.width, self.height
        new = [0] * (w * h)
        for y in range(h):
            for x in range(w):
                src_y = y - lines
                if src_y >= 0:
                    new[y*w + x] = self.display[src_y*w + x]
                else:
                    new[y*w + x] = 0
        self.display = new

# ==============================
# Pygame Frontend
# ==============================
class Frontend:
    def __init__(self, chip8: Chip8, scale: int = 10, tone_hz: int = 440):
        self.chip8 = chip8
        self.scale = max(1, int(scale))
        # ensure window fits even for high res
        width_px = chip8.width * self.scale
        height_px = chip8.height * self.scale
        self.surface = pygame.display.set_mode((width_px, height_px))
        pygame.display.set_caption("CHIP-8 / SCHIP (Python)")
        self.clock = pygame.time.Clock()

        # Audio setup (simple square tone)
        self.tone_hz = tone_hz
        self._init_audio()

    def _init_audio(self):
        try:
            pygame.mixer.pre_init(44100, -16, 1, 256)
            pygame.mixer.init()
        except Exception:
            return
        # generate a 100ms square wave buffer
        try:
            import numpy as np
        except Exception:
            self.sound = None
            return
        sr = 44100
        duration = 0.1
        t = np.arange(int(sr * duration))
        wave = ((t * self.tone_hz * 2 / sr) % 2 >= 1).astype('float32') * 2 - 1
        wave = (wave * 32767).astype('int16')
        self.sound = pygame.mixer.Sound(wave)
        self.sound.set_volume(0.2)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            elif event.type in (pygame.KEYDOWN, pygame.KEYUP):
                is_down = event.type == pygame.KEYDOWN
                # Escape to quit
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)
                # Map keys
                for k_idx, pgk in KEYMAP.items():
                    if event.key == pgk:
                        self.chip8.keys[k_idx] = is_down

    def render(self):
        # If resolution changed (SCHIP mode switch), recreate window
        desired_w = self.chip8.width * self.scale
        desired_h = self.chip8.height * self.scale
        cur = self.surface.get_size()
        if cur != (desired_w, desired_h):
            self.surface = pygame.display.set_mode((desired_w, desired_h))

        surf = self.surface
        surf.lock()
        surf.fill((0, 0, 0))
        pixel_size = self.scale
        for y in range(self.chip8.height):
            for x in range(self.chip8.width):
                if self.chip8.display[y * self.chip8.width + x]:
                    rect = pygame.Rect(x * pixel_size, y * pixel_size, pixel_size, pixel_size)
                    pygame.draw.rect(surf, (255, 255, 255), rect)
        surf.unlock()
        pygame.display.flip()

    def tick(self, fps: int):
        self.clock.tick(fps)

    def play_sound_if_needed(self):
        if hasattr(self, 'sound') and self.chip8.sound_timer > 0 and self.sound is not None:
            # Fire-and-forget short blip
            self.sound.play()

# ==============================
# Main loop
# ==============================

def main():
    parser = argparse.ArgumentParser(description="CHIP-8 / SCHIP emulator in Python")
    parser.add_argument("rom", help="Path to CHIP-8 ROM")
    parser.add_argument("--scale", type=int, default=8, help="Pixel scale factor (default 8)")
    parser.add_argument("--clock", type=int, default=700, help="CPU clock in Hz (default 700)")
    parser.add_argument("--legacy-store", action="store_true", help="Use original FX55/FX65 quirk (I increments)")
    parser.add_argument("--legacy-shift", action="store_true", help="Use original 8XY6/8XYE quirk (Vx = Vy first)")
    parser.add_argument("--alt-bnnn", action="store_true", help="Use alternative BNNN behavior (uses Vx as offset)")
    parser.add_argument("--superchip", action="store_true", help="Enable Super-CHIP features (128x64, extra opcodes)")
    parser.add_argument("--tone", type=int, default=440, help="Beep tone frequency in Hz")
    args = parser.parse_args()

    pygame.init()
    pygame.display.set_allow_screensaver(True)

    chip8 = Chip8(legacy_store=args.legacy_store, legacy_shift=args.legacy_shift, alt_bnnn=args.alt_bnnn)
    chip8.set_schip(args.superchip)

    # Load font
    chip8.memory[FONT_ADDRESS:FONT_ADDRESS + len(FONTSET)] = bytes(FONTSET)

    # Load ROM
    with open(args.rom, 'rb') as f:
        rom_data = f.read()
    chip8.load_rom(rom_data)

    frontend = Frontend(chip8, scale=args.scale, tone_hz=args.tone)

    # Timers tick at 60 Hz
    timer_hz = 60
    cycles_per_frame = max(1, args.clock // timer_hz)

    last_timer_tick = time.perf_counter()
    timer_period = 1.0 / timer_hz

    # Main emulation loop
    while True:
        frontend.handle_events()

        # Run CPU cycles for this frame
        for _ in range(cycles_per_frame):
            try:
                chip8.emulate_cycle()
            except NotImplementedError as e:
                print(e)
                pygame.quit()
                sys.exit(1)

        # Timer update at ~60 Hz
        now = time.perf_counter()
        if now - last_timer_tick >= timer_period:
            if chip8.delay_timer > 0:
                chip8.delay_timer -= 1
            if chip8.sound_timer > 0:
                chip8.sound_timer -= 1
            last_timer_tick = now

        # Optional sound beep when ST > 0
        frontend.play_sound_if_needed()

        # Render when needed (most games draw each frame anyway)
        if chip8.draw_flag:
            frontend.render()
            chip8.draw_flag = False

        # Cap UI thread to ~60 FPS for smoothness
        frontend.tick(60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(""
Exiting.")




