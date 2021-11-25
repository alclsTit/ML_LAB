import random
import pygame

import Object.EnvObject
import numpy as np
import random
from collections import deque

NUM_CHANNELS = 17
NUM_ACTIONS = 3


class UnitAction:
    MOVE_FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2


class UnitStateTransition:
    Dx, Dy = [-1, 0, 1, 0], [0, 1, 0, -1]

    def __init__(self, field_size, field, num_feed, init_pos, init_unit):
        self.field_height, self.field_width = field_size
        self.field = field.copy()
        self.ox, self.oy = init_pos
        self.dir = init_unit[-1]
        self.unit = deque(init_unit)

        for x in range(num_feed):
            self.generate_feed()

    def generate_feed(self):
        empty_blocks = []
        for i in range(self.field_height):
            for j in range(self.field_width):
                if self.field[i][j] == Object.EnvObject.EmptyBlock.get_code():
                    empty_blocks.append((i, j))

        if len(empty_blocks) > 0:
            x, y = random.sample(empty_blocks, 1)[0]
            self.field[x, y] = Object.EnvObject.FeedBlock.get_code()

    # NUM_CHANNELS X NUM_CAHNNELS 의 정방행렬 생성
    def get_state(self):
        return np.eye(NUM_CHANNELS)[self.field]

    def get_length(self):
        return len(self.unit) + 1

    def move_forward(self):
        ox = self.ox + UnitStateTransition.Dx[self.dir]
        oy = self.oy + UnitStateTransition.Dy[self.dir]

        if ox < 0 or oy < 0 or ox >= self.field_height or oy >= self.field_width \
                or Object.EnvObject.ObstacleBlock.contains(self.field[ox][oy]):
            return -1, True

        is_feed = Object.EnvObject.FeedBlock.contains(self.field[ox][oy])

        if not is_feed:
            self.field[self.ox, self.oy] = Object.EnvObject.EmptyBlock.get_code()
            td = self.unit.popleft()
            self.ox += UnitStateTransition.Dx[td]
            self.oy += UnitStateTransition.Dy[td]

        self.unit.append(self.dir)
        self.ox, self.oy = ox, oy

        if is_feed:
            self.generate_feed()
            return self.get_length(), False

        return 0, False

    def turn_left(self):
        self.dir = (self.dir + 3) % 4
        return self.move_forward()

    def turn_right(self):
        self.dir = (self.dir + 1) % 4
        return self.move_forward()



class Unit:
    ACTIONS = {
        UnitAction.MOVE_FORWARD: 'move_forward',
        UnitAction.TURN_LEFT: 'turn_left',
        UnitAction.TURN_RIGHT: 'turn_right'
    }

    def __init__(self, level_loader, block_pixels=30):
        self.total_reward = 0
        self.state_transition = 0
        self.level_loader = level_loader
        self.block_pixels = block_pixels

        self.field_height, self.field_width = self.level_loader.get_field_size()

        # pygame init -> display -> clock -> reset
        pygame.init()

        self.screen = pygame.display.set_mode((
            self.field_width * block_pixels,
            self.field_height * block_pixels
        ))
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.state_transition = UnitStateTransition(
            self.level_loader.get_field_size(),
            self.level_loader.get_field(),
            self.level_loader.get_num_feed(),
            self.level_loader.get_init_pos(),
            self.level_loader.get_init_unit()
        )

        self.total_reward = 0
        return self.state_transition.get_state()

    def step(self, action):
        reward, done = getattr(self.state_transition, Unit.ACTIONS[action])
        self.total_reward += reward
        return self.state_transition.get_state(), reward, done

    def get_length(self):
        return self.state_transition.get_length()

    def quit(self):
        pygame.quit()

    def render(self, fps):
        pygame.display.set_caption('caption')
        pygame.event.pump()
        self.screen.fill((255, 255, 255))

        for i in range(self.field_height):
            for j in range(self.field_width):
                point = get_color_point(self.state_transition.field[i][j])
                if point is None:
                    continue

                    pygame.draw.polygon(
                        self.screen,
                        point[0],
                        (point[1] + [j, i]) * self.block_pixels
                    )

        pygame.display.flip()
        self.clock.tick(fps)

    def save_image(self, save_path):
        pygame.image.save(self.screen, save_path)