from typing import *

from mlagents.envs import AllBrainInfo, UnityEnvironment
from mlagents.trainers import PolicyDef, Policy
from typing import *

from mlagents.envs import UnityEnvironment
from mlagents.trainers import PolicyDef, ActorManager, Experience
from multiprocessing import Process, Queue, Pipe, Manager
from multiprocessing.connection import Connection

from threading import Thread
from concurrent.futures import Future
import concurrent.futures as futures

from copy import deepcopy
import numpy as np


class EnvironmentCommand(NamedTuple):
    name: str
    worker_id: Optional[int] = None
    payload: Any = None


def worker(parent_conn: Connection, env_factory: Callable[[int], UnityEnvironment], worker_id: int):
    env = env_factory(worker_id)
    try:
        while True:
            cmd = parent_conn.recv()
            if cmd.name == 'step':
                vector_action, memory, text_action, value = cmd.payload
                all_brain_info = env.step(vector_action, memory, text_action, value)
                parent_conn.send(
                    EnvironmentCommand('step', worker_id, all_brain_info)
                )
            elif cmd.name == 'external_brains':
                parent_conn.send(
                    EnvironmentCommand('external_brains', worker_id, env.external_brains)
                )
            elif cmd.name == 'reset_parameters':
                parent_conn.send(
                    EnvironmentCommand('reset_parameters', worker_id, env.reset_parameters)
                )
            elif cmd.name == 'reset':
                all_brain_info = env.reset(cmd.payload[0], cmd.payload[1])
                parent_conn.send(
                    EnvironmentCommand('reset', worker_id, all_brain_info)
                )
            elif cmd.name == 'global_done':
                parent_conn.send(
                    EnvironmentCommand('global_done', worker_id, env.global_done)
                )
            elif cmd.send == 'close':
                env.close()
                break
    except KeyboardInterrupt:
        print('UnityEnvironment worker: keyboard interrupt')
    finally:
        env.close()


class UnityEnvWorkerInfo(NamedTuple):
    process: Process
    worker_id: int
    # inbox: Connection
    conn: Connection


class SubprocessUnityEnvironment:
    def __init__(self, env_factory: Callable[[int], UnityEnvironment], n_env: int = 1):
        self.envs = []
        self.env_agent_counts = {}
        self.waiting = False
        manager = Manager()
        for i in range(n_env):
            parent_conn, child_conn = Pipe()
            p = Process(target=worker, args=(child_conn, env_factory, i))
            p.start()
            self.envs.append(UnityEnvWorkerInfo(p, i, parent_conn))

    def step_async(self, vector_action=None, memory=None, text_action=None, value=None) -> None:
        if self.waiting:
            raise Exception('Tried to take step, but awaiting previous step.')

        agent_counts_cum = {}
        for brain_name in self.env_agent_counts.keys():
            agent_counts_cum[brain_name] = np.cumsum(self.env_agent_counts[brain_name])

        for worker_id, env in enumerate(self.envs):
            env_actions = {}
            env_memory = {}
            env_text_action = {}
            env_value = {}
            for brain_name in self.env_agent_counts.keys():
                start_ind = 0
                if worker_id > 0:
                    start_ind = agent_counts_cum[brain_name][worker_id - 1]
                end_ind = agent_counts_cum[brain_name][worker_id]
                if vector_action[brain_name] is not None:
                    env_actions[brain_name] = vector_action[brain_name][start_ind:end_ind]
                if memory[brain_name] is not None:
                    env_memory[brain_name] = memory[brain_name][start_ind:end_ind]
                if text_action[brain_name] is not None:
                    env_text_action[brain_name] = text_action[brain_name][start_ind:end_ind]
                if value[brain_name] is not None:
                    env_value[brain_name] = value[brain_name][start_ind:end_ind]
            env.conn.send(
                EnvironmentCommand('step', None, (env_actions, env_memory, env_text_action, env_value))
            )
        self.waiting = True

    def step_await(self) -> List[AllBrainInfo]:
        if not self.waiting:
            raise Exception('Tried to await, but step not taken.')

        steps = [self.envs[i].conn.recv().payload for i in range(len(self.envs))]
        self._get_agent_counts(steps)
        self.waiting = False
        return steps

    def _get_agent_counts(self, step_list):
        for i, step in enumerate(step_list):
            for brain_name, brain_info in step.items():
                if brain_name not in self.env_agent_counts.keys():
                    self.env_agent_counts[brain_name] = [0] * len(self.envs)
                self.env_agent_counts[brain_name][i] = len(brain_info.agents)

    def step(self, vector_action=None, memory=None, text_action=None, value=None) -> AllBrainInfo:
        self.step_async(vector_action, memory, text_action, value)
        steps = self.step_await()
        combined_brain_info = None
        for i, env in enumerate(self.envs):
            all_brain_info = steps[i]
            for brain_name, brain_info in all_brain_info.items():
                for j in range(len(brain_info.agents)):
                    brain_info.agents[j] = str(i) + '-' + str(brain_info.agents[j])
                if combined_brain_info:
                    combined_brain_info[brain_name].merge(brain_info)
            if not combined_brain_info:
                combined_brain_info = all_brain_info
        return combined_brain_info

    def reset(self, config=None, train_mode=True) -> AllBrainInfo:
        self._broadcast_message(EnvironmentCommand('reset', None, (config, train_mode)))
        reset_results = [self.envs[i].conn.recv() for i in range(len(self.envs))]
        self._get_agent_counts(map(lambda r: r.payload, reset_results))

        accumulated_brain_info = None
        for reset_result in reset_results:
            all_brain_info = reset_result.payload
            for brain_name, brain_info in all_brain_info.items():
                for i in range(len(brain_info.agents)):
                    brain_info.agents[i] = str(reset_result.worker_id) + '-' + str(brain_info.agents[i])
                if accumulated_brain_info:
                    accumulated_brain_info[brain_name].merge(brain_info)
            if not accumulated_brain_info:
                accumulated_brain_info = all_brain_info
        return accumulated_brain_info

    @property
    def global_done(self):
        self._broadcast_message(EnvironmentCommand('global_done'))
        dones: List[EnvironmentCommand] = [self.envs[i].conn.recv().payload for i in range(len(self.envs))]
        return all(dones)

    @property
    def external_brains(self):
        self.envs[0].conn.send(EnvironmentCommand('external_brains'))
        return self.envs[0].conn.recv().payload

    @property
    def reset_parameters(self):
        self.envs[0].conn.send(EnvironmentCommand('reset_parameters'))
        return self.envs[0].conn.recv().payload

    def close(self):
        for env in self.envs:
            env.process.join()

    def _broadcast_message(self, msg: EnvironmentCommand):
        for env in self.envs:
            env.conn.send(msg)
