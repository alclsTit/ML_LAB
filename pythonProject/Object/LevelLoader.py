import yaml as yam
import EnvObject
import numpy as np


class LevelLoader:
    def __init__(self, filepath):
        with open(filepath) as f:
            self.level = yam.safe_load(f)

        val = self.level['field']
        height, weight = len(val), len(val[0])
        self.field_size = height, weight

        # height x weigh 행렬 값을 0으로 채워넣기
        self.field = np.full(self.field_size, EnvObject.EmptyBlock.get_code())

        for i in range(height):
            for j in range(weight):
                if val[i][j] == '#':
                    self.field[i, j] = EnvObject.ObstacleBlock.get_code()
                elif val[i][j] == '@':
                    self.init_pos = i, j

        ox, oy = self.init_pos
        self.init_unit = []

        while True:
            next_dir, tx, ty = None, None, None



    def get_field_size(self):
        return self.field_size

    def get_field(self):
        return self.field

    def get_num_feed(self):
        return self.level['num_feed']

    def get_init_pos(self):
        return self.init_pos

    def get_init_unit(self):
        return self.init_unit