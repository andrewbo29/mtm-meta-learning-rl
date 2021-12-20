import torch

from src.replay_memory import ReplayMemory


class Runner:
    def __init__(self, env, device):
        self.env = env
        self.device = device
        self._needs_reset = True
        self._current_state = None

    def __del__(self):
        del self.env
        del self.device
        del self._needs_reset, self._current_state

    def reset(self):
        self._current_state = torch.tensor(self.env.reset()).float()
        self._needs_reset = False
        return self._current_state

    def run(self, get_action, episodes=100):
        replay = ReplayMemory(self.device)
        collected_episodes = 0
        collected_steps = 0

        i = 0
        while True:
            if collected_episodes >= episodes:
                replay.to_tensor()
                return replay

            if self._needs_reset:
                self.reset()

            action = get_action(self._current_state)
            action = action.squeeze()
            old_state = self._current_state

            state, reward, done, info = self.env.step(action.cpu().detach().numpy())
            success = info["success"]

            if self.env.curr_path_length >= self.env.max_path_length:
                done = True

            if done or success:
                collected_episodes += 1
                self._needs_reset = True

            state = torch.tensor(state).float()
            reward = torch.tensor([reward]).float()
            done = torch.tensor([done]).float()
            success = torch.tensor([success]).float()

            replay.push(old_state, action, reward, state, done, success)
            self._current_state = state

            i += 1
            collected_steps += 1
