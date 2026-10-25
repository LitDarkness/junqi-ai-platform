"""Multi-process sharding for the C++ batch self-play environment."""
from __future__ import annotations

import multiprocessing as mp
import random
import traceback
from multiprocessing.connection import wait

import numpy as np

from junqi_rl.env.cpp_selfplay_vec_env import CppSelfPlayVecEnv, ModelCache
from junqi_rl.training_config import OpponentPoolConfig, RewardConfig


def _worker(
    remote,
    num_envs: int,
    pool_dir: str,
    max_steps: int,
    reward_config: RewardConfig,
    opponent_config: OpponentPoolConfig,
    opponent_device: str,
    seed: int,
    remote_opponent_inference: bool,
):
    try:
        random.seed(seed)
        np.random.seed(seed)
        if not remote_opponent_inference:
            import torch
            torch.manual_seed(seed)
            torch.set_num_threads(1)
        def request_inference(model_path, features, masks):
            remote.send(("infer", (model_path, features, masks)))
            response, actions = remote.recv()
            if response != "infer_result":
                raise RuntimeError(f"Unexpected inference response: {response}")
            return actions

        env = CppSelfPlayVecEnv(
            num_envs=num_envs,
            pool_dir=pool_dir,
            max_steps=max_steps,
            reward_config=reward_config,
            opponent_config=opponent_config,
            opponent_device=opponent_device,
            opponent_inference=request_inference if remote_opponent_inference else None,
        )
        remote.send(("ready", None))
        while True:
            command, payload = remote.recv()
            if command == "reset":
                remote.send(("ok", env.reset()))
            elif command == "step":
                remote.send(("ok", env.step(payload)))
            elif command == "close":
                env.close()
                remote.send(("ok", None))
                break
            else:
                raise RuntimeError(f"Unknown shard command: {command}")
    except BaseException:
        try:
            remote.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        remote.close()


class ShardedCppSelfPlayVecEnv:
    """Run independent C++ batches concurrently and merge their NumPy results."""

    def __init__(
        self,
        num_envs: int,
        num_shards: int,
        pool_dir: str,
        max_steps: int,
        reward_config: RewardConfig,
        opponent_config: OpponentPoolConfig,
        opponent_device: torch.device,
        seed: int,
        centralize_opponent_inference: bool = True,
    ):
        if num_shards < 1 or num_envs % num_shards:
            raise ValueError("num_envs must be divisible by num_shards")
        self.num_envs = num_envs
        self.num_shards = num_shards
        self.shard_size = num_envs // num_shards
        self.closed = False
        self.centralize_opponent_inference = centralize_opponent_inference
        import torch
        self.torch = torch
        self.inference_device = opponent_device
        self.model_cache = ModelCache(opponent_device) if centralize_opponent_inference else None
        context = mp.get_context("spawn")
        self.remotes = []
        self.processes = []
        for shard in range(num_shards):
            parent, child = context.Pipe()
            process = context.Process(
                target=_worker,
                args=(
                    child,
                    self.shard_size,
                    pool_dir,
                    max_steps,
                    reward_config,
                    opponent_config,
                    str(opponent_device),
                    seed + shard * 1009,
                    centralize_opponent_inference,
                ),
                name=f"junqi-rollout-{shard}",
                daemon=True,
            )
            process.start()
            child.close()
            self.remotes.append(parent)
            self.processes.append(process)
        try:
            for index, remote in enumerate(self.remotes):
                if not remote.poll(90):
                    raise RuntimeError(f"Rollout shard {index} did not start within 90 seconds")
                self._receive(remote)
        except BaseException:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
            for process in self.processes:
                process.join(timeout=2)
            raise

    @staticmethod
    def _receive(remote):
        try:
            status, payload = remote.recv()
        except EOFError as exc:
            raise RuntimeError("Rollout shard exited without a response") from exc
        if status == "error":
            raise RuntimeError(f"Rollout shard failed:\n{payload}")
        return payload

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = self._collect_results()
        observations = np.concatenate([result[0] for result in results], axis=0)
        infos = tuple(info for result in results for info in result[1])
        return observations, infos

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int32)
        chunks = np.split(actions, self.num_shards)
        for remote, chunk in zip(self.remotes, chunks):
            remote.send(("step", chunk))
        results = self._collect_results()
        observations = np.concatenate([result[0] for result in results], axis=0)
        rewards = np.concatenate([result[1] for result in results])
        dones = np.concatenate([result[2] for result in results])
        infos = tuple(info for result in results for info in result[3])
        return observations, rewards, dones, infos

    def _collect_results(self):
        """Service nested GPU inference requests until every shard replies."""
        pending = set(self.remotes)
        completed = {}
        while pending:
            ready = wait(pending)
            inference_requests = []
            for remote in ready:
                try:
                    status, payload = remote.recv()
                except EOFError as exc:
                    raise RuntimeError("Rollout shard exited during an operation") from exc
                if status == "infer":
                    inference_requests.append((remote, payload))
                elif status == "ok":
                    completed[remote] = payload
                    pending.remove(remote)
                elif status == "error":
                    raise RuntimeError(f"Rollout shard failed:\n{payload}")
                else:
                    raise RuntimeError(f"Unknown rollout response: {status}")
            if inference_requests:
                self._serve_inference(inference_requests)
        return [completed[remote] for remote in self.remotes]

    def _serve_inference(self, requests):
        grouped = {}
        for remote, (model_path, features, masks) in requests:
            grouped.setdefault(model_path, []).append((remote, features, masks))
        for model_path, items in grouped.items():
            features = np.concatenate([item[1] for item in items], axis=0)
            masks = np.concatenate([item[2] for item in items], axis=0)
            model = self.model_cache.get(model_path)
            state = self.torch.from_numpy(features).to(self.inference_device)
            mask = self.torch.from_numpy(masks).to(self.inference_device, dtype=self.torch.bool)
            with self.torch.no_grad():
                actions, _, _, _ = model.get_action_and_value(state, mask, deterministic=False)
            actions = actions.cpu().numpy()
            offset = 0
            for remote, item_features, _ in items:
                count = item_features.shape[0]
                remote.send(("infer_result", actions[offset:offset + count]))
                offset += count

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for remote in self.remotes:
            try:
                self._receive(remote)
            except (BrokenPipeError, EOFError, OSError, RuntimeError):
                pass
            remote.close()
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        self.closed = True

    def __del__(self):
        self.close()
