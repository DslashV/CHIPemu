
import pygame
import tkinter as tk
from tkinter import filedialog
import random

# ======================================================
# Chip-8 Emulator
# ======================================================
class Chip8:
    def __init__(self):
        self.reset()

    def reset(self):
        # Memory and CPU registers
        self.memory = [0] * 4096
        self.V = [0] * 16
        self.I = 0
        self.pc = 0x200

        # Graphics (64x32)
        self.gfx = [0] * (64*32)

        # Timers
        self.delay_timer = 0
        self.sound_timer = 0

        # Stack and keypad
        self.stack = []
        self.keys = [0] * 16

        # Flags
        self.draw_flag = True  # Force redraw after reset
        self.running = False

        # Load fontset
        fontset = [
            0xF0,0x90,0x90,0x90,0xF0, 0x20,0x60,0x20,0x20,0x70,
            0xF0,0x10,0xF0,0x80,0xF0, 0xF0,0x10,0xF0,0x10,0xF0,
            0x90,0x90,0xF0,0x10,0x10, 0xF0,0x80,0xF0,0x10,0xF0,
            0xF0,0x80,0xF0,0x90,0xF0, 0xF0,0x10,0x20,0x40,0x40,
            0xF0,0x90,0xF0,0x90,0xF0, 0xF0,0x90,0xF0,0x10,0xF0,
            0xF0,0x90,0xF0,0x90,0x90, 0xE0,0x90,0xE0,0x90,0xE0,
            0xF0,0x80,0x80,0x80,0xF0, 0xE0,0x90,0x90,0x90,0xE0,
            0xF0,0x80,0xF0,0x80,0xF0, 0xF0,0x80,0xF0,0x80,0x80
        ]
        for i, byte in enumerate(fontset):
            self.memory[i] = byte

    def load_rom(self, rom_data):
        self.reset()
        size = min(len(rom_data), 4096-0x200)
        for i in range(size):
            self.memory[0x200 + i] = rom_data[i]
        self.running = True
        print(f"[INFO] ROM loaded ({size} bytes)!")

    def execute_cycle(self):
        if not self.running or self.pc >= 4094:
            return

        opcode = self.memory[self.pc]<<8 | self.memory[self.pc+1]
        self.pc += 2

        nnn = opcode & 0x0FFF
        n   = opcode & 0x000F
        x   = (opcode & 0x0F00) >> 8
        y   = (opcode & 0x00F0) >> 4
        kk  = opcode & 0x00FF

        # ==============================
        # Decode & Execute Opcodes
        # ==============================
        if opcode == 0x00E0:  # Clear screen
            self.gfx = [0]*(64*32)
            self.draw_flag=True
        elif opcode == 0x00EE:  # Return
            if self.stack:
                self.pc=self.stack.pop()
            else:
                print("[ERROR] Stack underflow!")
                self.running=False
        elif opcode & 0xF000 == 0x1000:  # Jump
            self.pc = nnn
        elif opcode & 0xF000 == 0x2000:  # Call subroutine
            self.stack.append(self.pc)
            self.pc = nnn
        elif opcode & 0xF000 == 0x3000:  # Skip if Vx == kk
            if self.V[x]==kk: self.pc+=2
        elif opcode & 0xF000 == 0x4000:  # Skip if Vx != kk
            if self.V[x]!=kk: self.pc+=2
        elif opcode & 0xF000 == 0x5000:  # Skip if Vx == Vy
            if self.V[x]==self.V[y]: self.pc+=2
        elif opcode & 0xF000 == 0x6000:  # LD Vx, kk
            self.V[x]=kk
        elif opcode & 0xF000 == 0x7000:  # ADD Vx, kk
            self.V[x]=(self.V[x]+kk)&0xFF
        elif opcode & 0xF000 == 0x8000:
            if n==0x0: self.V[x]=self.V[y]
            elif n==0x1: self.V[x]|=self.V[y]
            elif n==0x2: self.V[x]&=self.V[y]
            elif n==0x3: self.V[x]^=self.V[y]
            elif n==0x4:
                total=self.V[x]+self.V[y]
                self.V[0xF]=1 if total>0xFF else 0
                self.V[x]=total&0xFF
            elif n==0x5:
                self.V[0xF]=1 if self.V[x]>self.V[y] else 0
                self.V[x]=(self.V[x]-self.V[y])&0xFF
            elif n==0x6:
                self.V[0xF]=self.V[x]&0x1
                self.V[x]>>=1
            elif n==0x7:
                self.V[0xF]=1 if self.V[y]>self.V[x] else 0
                self.V[x]=(self.V[y]-self.V[x])&0xFF
            elif n==0xE:
                self.V[0xF]=(self.V[x]&0x80)>>7
                self.V[x]=(self.V[x]<<1)&0xFF
        elif opcode & 0xF000 == 0x9000:      # SNE Vx,Vy
            if self.V[x]!=self.V[y]: self.pc+=2
        elif opcode & 0xF000 == 0xA000:      # LD I, nnn
            self.I=nnn
        elif opcode & 0xF000 == 0xB000:      # JP V0+nnn
            self.pc=nnn+self.V[0]
        elif opcode & 0xF000 == 0xC000:      # RND Vx, kk
            self.V[x]=random.randint(0,255)&kk
        elif opcode & 0xF000 == 0xD000:      # DRW Vx,Vy,n
            vx,vy=self.V[x],self.V[y]
            height=n
            self.V[0xF]=0
            for row in range(height):
                if vy+row>=32: continue
                byte=self.memory[self.I+row]
                for col in range(8):
                    if vx+col>=64: continue
                    pixel=(byte>>(7-col))&1
                    idx=(vy+row)*64+(vx+col)
                    if pixel:
                        if self.gfx[idx]==1: self.V[0xF]=1
                        self.gfx[idx]^=1
            self.draw_flag=True
        elif opcode & 0xF000 == 0xE000:
            if kk==0x9E:
                if self.keys[self.V[x]]: self.pc+=2
            elif kk==0xA1:
                if not self.keys[self.V[x]]: self.pc+=2
        elif opcode & 0xF000 == 0xF000:
            if kk==0x07: self.V[x]=self.delay_timer
            elif kk==0x0A:
                pressed=False
                for i,val in enumerate(self.keys):
                    if val:
                        self.V[x]=i
                        pressed=True
                        break
                if not pressed: self.pc-=2
            elif kk==0x15: self.delay_timer=self.V[x]
            elif kk==0x18: self.sound_timer=self.V[x]
            elif kk==0x1E: self.I=(self.I+self.V[x])&0xFFF
            elif kk==0x29: self.I=self.V[x]*5
            elif kk==0x33:
                self.memory[self.I]=self.V[x]//100
                self.memory[self.I+1]=(self.V[x]%100)//10
                self.memory[self.I+2]=self.V[x]%10
            elif kk==0x55:
                for i in range(x+1): self.memory[self.I+i]=self.V[i]
            elif kk==0x65:
                for i in range(x+1): self.V[i]=self.memory[self.I+i]
        else:
            print(f"[WARN] Unknown opcode {opcode:04X}")

        # Timers
        if self.delay_timer>0: self.delay_timer-=1
        if self.sound_timer>0: self.sound_timer-=1

    def draw_screen(self, surface, scale=10):
        if self.draw_flag:
            surface.fill((0,0,0))
            for y in range(32):
                for x in range(64):
                    if self.gfx[y*64+x]:
                        pygame.draw.rect(surface,(255,255,255),(x*scale,y*scale,scale,scale))
            self.draw_flag=False

# ======================================================
# File picker
# ======================================================
def open_rom_file():
    root=tk.Tk()
    root.withdraw()
    path=filedialog.askopenfilename(title="Select Chip-8 ROM",
                                    filetypes=[("Chip-8 ROMs","*.ch8 *.c8 *.rom"),("All files","*.*")])
    if not path: return None
    with open(path,"rb") as f:
        return f.read()

# ======================================================
# Key mapping
# ======================================================
KEY_MAP = {
    pygame.K_x:0x0, pygame.K_1:0x1, pygame.K_2:0x2, pygame.K_3:0x3,
    pygame.K_q:0x4, pygame.K_w:0x5, pygame.K_e:0x6,
    pygame.K_a:0x7, pygame.K_s:0x8, pygame.K_d:0x9,
    pygame.K_z:0xA, pygame.K_c:0xB, pygame.K_4:0xC,
    pygame.K_r:0xD, pygame.K_f:0xE, pygame.K_v:0xF
}

# ======================================================
# Main loop
# ======================================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((64*10,32*10))
    pygame.display.set_caption("Chip-8 Emulator - F10 load ROM, F9 reset")
    clock = pygame.time.Clock()

    chip8 = Chip8()

    print("[INFO] Emulator started! Press F10 to load a ROM or drag & drop a file. F9 fully resets.")

    running=True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running=False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F10:
                    rom = open_rom_file()
                    if rom:
                        chip8.load_rom(rom)
                elif event.key == pygame.K_F9:
                    chip8.reset()  # Fully reset, do NOT reload ROM
                    print("[INFO] Emulator fully reset!")
                elif event.key in KEY_MAP:
                    chip8.keys[KEY_MAP[event.key]] = 1
            elif event.type == pygame.KEYUP:
                if event.key in KEY_MAP:
                    chip8.keys[KEY_MAP[event.key]] = 0
            elif event.type == pygame.DROPFILE:
                with open(event.file,"rb") as f:
                    rom = f.read()
                chip8.load_rom(rom)

        if chip8.running:
            chip8.execute_cycle()
            chip8.draw_screen(screen)

        pygame.display.flip()
        clock.tick(500)

    pygame.quit()

if __name__=="__main__":
    main()
