import numpy as np


class Block:
    @staticmethod
    def get_code(**args):
        pass

    @staticmethod
    def contains(**args):
        pass

    @staticmethod
    def get_color(**args):
        pass

    @staticmethod
    def get_point(**args):
        pass


class EmptyBlock(Block):
    @staticmethod
    def get_code():
        return 0

    @staticmethod
    def contains(code):
        return code == 0


class ObstacleBlock(Block):
    @staticmethod
    def get_code():
        return 1

    @staticmethod
    def contains(code):
        return code == 1

    @staticmethod
    def get_color():
        return 255, 255, 255

    @staticmethod
    def get_point():
        return np.array([0, 0], [1, 0], [1, 1], [0, 1])


class FeedBlock(Block):
    @staticmethod
    def get_code():
        return 2

    @staticmethod
    def contains(code):
        return code == 2

    @staticmethod
    def get_color():
        return 255, 0, 0

    @staticmethod
    def get_point():
        return np.array([0.4, 0.2], [0.6, 0.2], [0.8, 0.4], [0.8, 0.6],
                        [0.6, 0.8], [0.4, 0.8], [0.2, 0.6], [0.2, 0.4])
