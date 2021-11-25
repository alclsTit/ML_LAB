import os
import random
import Object.LevelLoader as LevelLoader

import Unit
import DQNAgent

class DQNTrainer:
    def __init__(self,
        level_filepath,
        max_step=2000,
        episodes = 30000,
        init_epsilon = 1,
        min_epsilon = 0.1,
        exploration_ratio = 0.5,
        save_dir='checkpoints',
        save_freq = 500,
        batch_size = 64,
        gamma = 0.99,
        seed = 42,
        enable_save = True,
        enable_render = True,
        render_freq = 500,
        render_fps = 20,
        min_replay_memory_size = 1000,
        replay_memory_size = 100000,
        target_update_freq = 5):

        self.set_random_seed(seed)
        self.episodes = episodes
        self.max_steps = max_step
        self.epsilon = init_epsilon
        self.min_epsilon = min_epsilon
        self.exploration_ratio = exploration_ratio,
        self.render_freq = render_freq,
        self.enable_render = enable_render,
        self.render_fps = render_fps,
        self.save_dir = save_dir,
        self.enable_save = enable_save,
        self.save_freq = save_freq

        if enable_save and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        level_loader = LevelLoader(level_filepath)

        self.agent = DQNAgent(
            level_loader.get_field_size(),
            gamma = gamma,
            batch_size = batch_size,
            min_replay_memory_size = min_replay_memory_size,
            replay_memory_size = replay_memory_size,
            target_update_freq = target_update_freq
        )

        self.env = Unit(level_loader)
        self.summary = Summary()
        self.cur_episodes = 0
        self.