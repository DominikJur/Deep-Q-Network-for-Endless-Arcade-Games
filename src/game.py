import os
import random
import sys

import pygame
from pygame import mixer

from src.coin import Coin
from src.player import Player
from src.spike import (Ceilling_Spike, East_Wall_Spike, Floor_Spike,
                       West_Wall_Spike)


class Game:
    def __init__(self, headless=False, player_id=None, player_color=None):
        if not pygame.get_init():
            pygame.init()
            
        self.headless = headless
        self.scale = 2
        self.user_x = 288 * self.scale
        self.user_y = 352 * self.scale
        self.spike_count_override = None

        if not self.headless:
            self.screen = pygame.display.set_mode((self.user_x, self.user_y))
            pygame.display.set_caption("Beware of the skewers!")
        else:
            self.screen = pygame.display.set_mode((self.user_x, self.user_y))

        if player_color is None:
            self.player_color = (255, 255, 255)
        else:
            self.player_color = player_color
        if player_id is None:
            player_id = 0

        r, g, b = self.player_color
        self.player_color = (r, g, b, 255)

        self.player = Player(
            self.scale, self.user_x, self.user_y, self.player_color, player_id, headless=self.headless
        )

        self.high_score = self._load_high_score()
        self.start_high_score = self.high_score
        self.coin_total = self._load_coin_total()

        self.running = False
        self.score = 0

        self.player = Player(
            self.scale, self.user_x, self.user_y, self.player_color, player_id, headless=self.headless
        )
        self.east_spikes = []
        self.west_spikes = []
        self.coin_list = []

        if not self.headless:
            try:
                self.coin_sound = mixer.Sound(os.path.join("audio", "coin_collect.mp3"))
            except:
                self.coin_sound = None
        else:
            self.coin_sound = None

        self.clock = pygame.time.Clock()

        self._setup_floor_and_ceiling()

    def cleanup(self):
        try:
            if hasattr(self, 'coin_sound') and self.coin_sound:
                del self.coin_sound
            if hasattr(self, 'player'):
                del self.player
            if hasattr(self, 'east_spikes'):
                del self.east_spikes
            if hasattr(self, 'west_spikes'):
                del self.west_spikes
            if hasattr(self, 'coin_list'):
                del self.coin_list
            if hasattr(self, 'floor_spikes'):
                del self.floor_spikes
            if hasattr(self, 'ceiling_spikes'):
                del self.ceiling_spikes
        except:
            pass

    def _load_high_score(self):
        try:
            with open(os.path.join("data", "high_score.txt")) as file:
                return file.read().strip()
        except FileNotFoundError:
            return "0"

    def _load_coin_total(self):
        try:
            with open(os.path.join("data", "total_coins.txt")) as file:
                return int(file.read().strip())
        except FileNotFoundError:
            return 0

    def _save_data(self):
        try:
            if int(self.start_high_score) < int(self.high_score):
                os.makedirs("data", exist_ok=True)
                with open(os.path.join("data", "high_score.txt"), "w") as file:
                    file.write(f"{self.high_score}")
            os.makedirs("data", exist_ok=True)
            with open(os.path.join("data", "total_coins.txt"), "w") as file:
                file.write(f"{self.coin_total}")
        except Exception as e:
            print(f"Error saving data: {e}")

    def _setup_floor_and_ceiling(self):
        self.floor_rect = pygame.Rect(
            0, round(self.user_y * 0.9), self.user_x, self.user_y
        )
        self.floor_spikes = [Floor_Spike(self.scale) for _ in range(9)]
        self.floor_spikes = [
            spike.set_position(
                i * self.scale * 32, round(self.user_y * 0.9) - spike.rect.height
            )
            for i, spike in enumerate(self.floor_spikes)
        ]

        self.ceiling_rect = pygame.Rect(
            0, -self.user_y, self.user_x, self.user_y + round(self.user_y * 0.1)
        )
        self.ceiling_spikes = [Ceilling_Spike(self.scale) for _ in range(9)]
        self.ceiling_spikes = [
            spike.set_position(i * self.scale * 32, round(self.user_y * 0.1))
            for i, spike in enumerate(self.ceiling_spikes)
        ]

    def draw_text(self, text, size, x, y):
        if self.headless:
            return
        try:
            font = pygame.font.Font("font\\Pixeltype.ttf", size)
            text_surface = font.render(text, True, "white")
            text_rect = text_surface.get_rect()
            text_rect.midtop = (x, y)
            self.screen.blit(text_surface, text_rect)
        except:
            font = pygame.font.Font(None, size)
            text_surface = font.render(text, True, "white")
            text_rect = text_surface.get_rect()
            text_rect.midtop = (x, y)
            self.screen.blit(text_surface, text_rect)

    def update_high_score(self):
        if int(self.high_score) < int(self.score):
            self.high_score = str(self.score)

    def _randomize_spikes(self, flag):
        if flag == 0:
            return [], [], 0

        if self.spike_count_override is not None:
            count = self.spike_count_override
        else:
            if self.score < 5:
                count = 1
            elif self.score < 10:
                count = 2
            elif self.score < 20:
                count = 3
            else:
                count = 4

        if flag == 1:
            return [East_Wall_Spike(self.scale) for _ in range(count)], [], count
        else:
            return [], [West_Wall_Spike(self.scale) for _ in range(count)], count

    def generate_spikes(self):
        east_spikes, west_spikes, count = self._randomize_spikes(
            self.player.check_wall_collision()
        )
        pos_list = random.sample(range(2, 9), count)

        west_spikes = [
            spike.set_position(0, pos_list.pop() * 32 * self.scale)
            for spike in west_spikes
        ]
        east_spikes = [
            spike.set_position(
                self.user_x - spike.rect.width, pos_list.pop() * 32 * self.scale
            )
            for spike in east_spikes
        ]

        return east_spikes, west_spikes

    def manage_spikes(self):
        for spike in self.east_spikes + self.west_spikes:
            if not self.headless:
                self.screen.blit(spike.image, spike.rect)
            if spike.rect.colliderect(self.player.rect):
                self.player.death()

    def generate_coins(self):
        if not self.coin_list:
            coin = Coin(self.scale)
            x = random.randint(round(0.2 * self.user_x), round(0.8 * self.user_x))
            y = random.randint(round(0.2 * self.user_y), round(0.8 * self.user_y))
            self.coin_list = [coin.set_position(x, y)]

    def manage_coins(self):
        for coin in self.coin_list:
            if not self.headless:
                self.screen.blit(coin.image, coin.rect)
            if coin.rect.colliderect(self.player.rect):
                if self.coin_sound:
                    self.coin_sound.play()
                self.coin_list = []
                self.coin_total += 1

    def check_static_spike_collisions(self):
        for spike in self.floor_spikes + self.ceiling_spikes:
            if spike.rect.colliderect(self.player.rect):
                self.player.death()

    def handle_wall_collision(self):
        if self.player.check_wall_collision() != 0:
            self.player.after_collision()
            self.score += 1
            self.generate_coins()
            self.east_spikes, self.west_spikes = self.generate_spikes()

    def reset_game(self):
        self.player.default_pos()
        self.player.dead = False
        self.coin_list = []
        self.score = 0
        self.east_spikes = []
        self.west_spikes = []
        self.running = False

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._save_data()
                if not self.headless:
                    pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN and event.key == self.player.jump_key:
                if self.player.dead:
                    self.reset_game()
                elif not self.running:
                    self.running = True
                    self.player.jump()
                else:
                    self.player.jump()

            if self.player.dead and event.type == pygame.MOUSEBUTTONDOWN:
                button_x = self.user_x // 3
                button_y = self.user_y // 5 * 2
                button_width = self.user_x // 3
                button_height = self.user_y // 6

                if (
                    button_x < event.pos[0] < button_x + button_width
                    and button_y < event.pos[1] < button_y + button_height
                ):
                    self.reset_game()

    def draw_background(self):
        if self.headless:
            return
            
        self.screen.fill("#c3c3c3")

        pygame.draw.rect(self.screen, "#5c5c5c", self.floor_rect)
        for spike in self.floor_spikes:
            self.screen.blit(spike.image, spike.rect)

        pygame.draw.rect(self.screen, "#5c5c5c", self.ceiling_rect)
        for spike in self.ceiling_spikes:
            self.screen.blit(spike.image, spike.rect)

    def draw_ui(self):
        if self.headless:
            return
            
        self.draw_text(
            f"Score: {self.score}",
            40 * self.scale,
            self.user_x // 2,
            self.user_y // 3 - 12 * self.scale,
        )
        self.draw_text(
            f"High Score: {self.high_score}",
            40 * self.scale,
            self.user_x // 2,
            self.user_y // 5 - 12 * self.scale,
        )
        self.draw_text(
            f"Coins: {self.coin_total}",
            40 * self.scale,
            self.user_x // 2,
            self.user_y // 4 * 3,
        )

    def draw_start_screen(self):
        if self.headless:
            return
            
        self.player.default_pos()
        self.player.start_screen_animation()
        self.draw_text(
            "Press jump key to start",
            40 * self.scale,
            self.user_x // 2,
            self.user_y // 2 + 30 * self.scale,
        )

    def draw_death_screen(self):
        if self.headless:
            return
            
        button_rect = pygame.Rect(
            self.user_x // 3, self.user_y // 5 * 2, self.user_x // 3, self.user_y // 6
        )
        pygame.draw.rect(self.screen, "#fd5f5f", button_rect, 90, 30)
        pygame.draw.rect(self.screen, "#bf5959", button_rect, 10, 30)
        self.draw_text(
            "Retry",
            40 * self.scale,
            self.user_x // 2,
            self.user_y // 2 - 16 * self.scale,
        )

    def update_game(self):
        self.check_static_spike_collisions()
        self.handle_wall_collision()
        self.manage_coins()
        self.manage_spikes()
        self.player.update()

        if self.player.dead:
            self.west_spikes = []
            self.east_spikes = []
            self.running = False

    def get_state_info(self):
        player_velocity = self.player.velocity
        player_pos = (self.player.rect.x, self.player.rect.y)
        player_gravity = self.player.gravity
        player_dead = self.player.dead
        east_spike_pos = [spike.rect.y for spike in self.east_spikes]
        west_spike_pos = [spike.rect.y for spike in self.west_spikes]

        return {
            "player_velocity": player_velocity,
            "player_pos": player_pos,
            "player_gravity": player_gravity,
            "player_dead": player_dead,
            "east_spike_pos": east_spike_pos,
            "west_spike_pos": west_spike_pos,
            "score": self.score,
            "coins": self.coin_total,
        }

    def run(self):
        if not self.headless:
            while True:
                self.handle_events()
                self.draw_background()
                self.draw_ui()

                if not self.player.dead:
                    if self.running:
                        self.update_game()
                    else:
                        self.draw_start_screen()
                else:
                    self.draw_death_screen()

                self.screen.blit(self.player.image, self.player.rect)

                self.update_high_score()

                pygame.display.update()
                self.clock.tick(30)
        else:
            while True:
                self.handle_events()

                if not self.player.dead:
                    if self.running:
                        self.update_game()
                    else:
                        self.draw_start_screen()
                else:
                    self.draw_death_screen()

                self.update_high_score()

                self.clock.tick(30)