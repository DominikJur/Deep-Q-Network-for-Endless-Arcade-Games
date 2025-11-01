import os

import pygame


class Coin(pygame.sprite.Sprite):
    def __init__(self, scale):
        super(Coin, self).__init__()
        self.image = pygame.image.load(os.path.join("graphics", "coin.png"))
        self.image = pygame.Surface.convert_alpha(self.image)
        self.image = pygame.transform.rotozoom(self.image, 0, scale)
        self.rect = pygame.Surface.get_rect(self.image)
        self.rect.x = 0
        self.rect.y = 0

    def set_position(self, x, y):
        self.rect.x = x
        self.rect.y = y
        return self
