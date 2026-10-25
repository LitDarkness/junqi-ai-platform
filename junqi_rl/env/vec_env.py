import numpy as np


class SubprocVecEnv:
    def __init__(self, env_fns):
        self.envs = [env_fn() for env_fn in env_fns]
        self.num_envs = len(self.envs)
        self.closed = False
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

    def reset(self):
        observations = []
        infos = []
        self._episode_returns.fill(0.0)
        self._episode_lengths.fill(0)
        for env in self.envs:
            obs, info = env.reset()
            observations.append(obs)
            infos.append(info)
        return np.stack(observations), tuple(infos)

    def step(self, actions):
        observations = []
        rewards = []
        dones = []
        infos = []

        for index, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

            self._episode_returns[index] += float(reward)
            self._episode_lengths[index] += 1

            if done:
                info = dict(info)
                info["episode"] = {
                    "r": float(self._episode_returns[index]),
                    "l": int(self._episode_lengths[index]),
                }
                obs, reset_info = env.reset()
                info["action_mask"] = reset_info["action_mask"]
                self._episode_returns[index] = 0.0
                self._episode_lengths[index] = 0

            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return (
            np.stack(observations),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.bool_),
            tuple(infos),
        )

    def close(self):
        if self.closed:
            return
        for env in self.envs:
            env.close()
        self.closed = True
