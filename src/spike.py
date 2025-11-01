import os

import pygame


class East_Wall_Spike(pygame.sprite.Sprite):
    def __init__(self, scale):
        super(East_Wall_Spike, self).__init__()
        self.image = pygame.image.load(os.path.join("graphics", "spike_east.png"))
        self.image = pygame.Surface.convert_alpha(self.image)
        self.image = pygame.transform.rotozoom(self.image, 0, scale)
        self.rect = pygame.Surface.get_rect(self.image)
        self.rect.x = 0
        self.rect.y = 0

    def set_position(self, x, y):
        self.rect.x = x
        self.rect.y = y
        return self


class West_Wall_Spike(pygame.sprite.Sprite):
    def __init__(self, scale):
        super(West_Wall_Spike, self).__init__()
        self.image = pygame.image.load(os.path.join("graphics", "spike_west.png"))
        self.image = pygame.Surface.convert_alpha(self.image)
        self.image = pygame.transform.rotozoom(self.image, 0, scale)
        self.rect = pygame.Surface.get_rect(self.image)
        self.rect.x = 0
        self.rect.y = 0

    def set_position(self, x, y):
        self.rect.x = x
        self.rect.y = y
        return self


class Ceilling_Spike(pygame.sprite.Sprite):
    def __init__(self, scale):
        super(Ceilling_Spike, self).__init__()
        self.image = pygame.image.load(os.path.join("graphics", "spike_north.png"))
        self.image = pygame.Surface.convert_alpha(self.image)
        self.image = pygame.transform.rotozoom(self.image, 0, scale)
        self.rect = pygame.Surface.get_rect(self.image)
        self.rect.x = 0
        self.rect.y = 0

    def set_position(self, x, y):
        self.rect.x = x
        self.rect.y = y
        return self


class Floor_Spike(pygame.sprite.Sprite):
    def __init__(self, scale):
        super(Floor_Spike, self).__init__()
        self.image = pygame.image.load(os.path.join("graphics", "spike_south.png"))
        self.image = pygame.Surface.convert_alpha(self.image)
        self.image = pygame.transform.rotozoom(self.image, 0, scale)
        self.rect = pygame.Surface.get_rect(self.image)
        self.rect.x = 0
        self.rect.y = 0

    def set_position(self, x, y):
        self.rect.x = x
        self.rect.y = y
        return self
