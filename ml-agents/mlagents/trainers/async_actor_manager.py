from typing import *

from mlagents.envs import AllBrainInfo, UnityEnvironment
from mlagents.trainers import PolicyDef, Policy
from typing import *

from mlagents.envs import UnityEnvironment
from mlagents.trainers import PolicyDef, ActorManager, Experience
from multiprocessing import Process, Queue, Manager

from threading import Thread
from concurrent.futures import Future
import concurrent.futures as futures

from copy import deepcopy


def call_with_future(fn, future, args, kwargs):
    try:
        result = fn(*args, **kwargs)
        future.set_result(result)
    except Exception as exc:
        future.set_exception(exc)


def threaded(fn):
    def wrapper(*args, **kwargs):
        future = Future()
        Thread(target=call_with_future, args=(fn, future, args, kwargs)).start()
        return future
    return wrapper


class AsyncActorManager:
    def __init__(self, env_factory: Callable[[int], UnityEnvironment], n_env: int=1):
        self.env_factory = env_factory
        env_futures: List[Future] = [self._start_env_async(i) for i in range(n_env)]
        self.envs: List[UnityEnvironment] = [f.result() for f in env_futures]
        self.latest_experience = None
        self._default_reset_params = None
        self.policies = {}

    @threaded
    def _start_env_async(self, worker_id: int):
        return self.env_factory(worker_id)

    def get_external_brains(self):
        return self.envs[0].external_brains

    def get_reset_parameters(self):
        return self.envs[0].reset_parameters

    def set_policies(self, policies: Dict[str, Policy]):
        self.policies = policies

    def set_default_reset_params(self, reset_params: Dict[str, str]):
        self._default_reset_params = reset_params

    def load_policies(self, serialized_policies: Dict[str, PolicyDef]):
        self.policies = {}
        for brain_name, policy_def in serialized_policies.items():
            self.policies[brain_name] = Policy.load_from_policy_def(policy_def, self.get_external_brains()[brain_name])

    @threaded
    def reset_envs_async(self, worker_id: int, config=None):
        return self.envs[worker_id].reset(config)

    @threaded
    def step_envs_async(self, worker_id: int, actions, memories, text, value):
        return self.envs[worker_id].step(actions, memories, text, value)

    def reset(self, config=None) -> Experience:
        reset_params = {}
        if self._default_reset_params is not None and config is not None:
            reset_params = self._default_reset_params.copy()
            reset_params.update(config)
        elif self._default_reset_params is not None:
            reset_params = self._default_reset_params
        elif config is not None:
            reset_params = config

        worker_brain_info_futures = []
        for worker_id in range(len(self.envs)):
            worker_brain_info_futures.append(
                self.reset_envs_async(worker_id, reset_params)
            )
        exp = self._combine_brain_infos(worker_brain_info_futures)
        self.latest_experience = exp
        return exp

    def _combine_brain_infos(self, worker_brain_info_futures, take_action_outputs=None):
        worker_brain_infos = [deepcopy(f.result()) for f in worker_brain_info_futures]
        all_combined_brain_info = None
        for worker_id, brain_infos in enumerate(worker_brain_infos):
            for brain_name, brain_info in brain_infos.items():
                for i in range(len(brain_info.agents)):
                    brain_info.agents[i] = str(worker_id) + '-' + str(brain_info.agents[i])
                if all_combined_brain_info:
                    all_combined_brain_info[brain_name].merge(brain_info)
            if not all_combined_brain_info:
                all_combined_brain_info = brain_infos
        return Experience(all_combined_brain_info, take_action_outputs)

    def is_global_done(self):
        dones = [env.global_done for env in self.envs]
        return all(dones)

    def close(self):
        [env.close() for env in self.envs]

    def take_step(self) -> Experience:
        # Decide and take an action
        take_action_vector = {}
        take_action_memories = {}
        take_action_text = {}
        take_action_value = {}
        take_action_outputs = {}
        for brain_name, policy in self.policies.items():
            action_info = policy.get_action(self.latest_experience.brain_info[brain_name])
            take_action_vector[brain_name] = action_info.action
            take_action_memories[brain_name] = action_info.memory
            take_action_text[brain_name] = action_info.text
            take_action_value[brain_name] = action_info.value
            take_action_outputs[brain_name] = action_info.outputs

        step_info_by_worker = []
        for i in range(len(self.envs)):
            step_info_by_worker.append(({}, {}, {}))

        for brain_name, brain_info in self.latest_experience.brain_info.items():
            agents_by_worker = [0] * len(self.envs)
            all_agents = brain_info.agents
            split_agent_strings = [a.split('-') for a in all_agents]
            for sas in split_agent_strings:
                agents_by_worker[int(sas[0])] += 1

            curr_index = 0
            for worker_id in range(len(self.envs)):
                worker_agents = agents_by_worker[worker_id]
                end = curr_index + worker_agents
                actions = take_action_vector[brain_name][curr_index:end]
                memories = None
                value = None
                if take_action_memories[brain_name] is not None:
                    memories = take_action_memories[brain_name][curr_index:end]
                if take_action_value[brain_name] is not None:
                    value = take_action_value[brain_name][curr_index:end]
                step_info_by_worker[worker_id][0][brain_name] = actions
                step_info_by_worker[worker_id][1][brain_name] = memories
                step_info_by_worker[worker_id][2][brain_name] = value
                curr_index = end

        worker_step_futures = []
        for worker_id, env in enumerate(self.envs):
            worker_step_futures.append(self.step_envs_async(
                worker_id,
                step_info_by_worker[worker_id][0],
                step_info_by_worker[worker_id][1],
                None,
                step_info_by_worker[worker_id][2]
            ))
        exp = self._combine_brain_infos(worker_step_futures, take_action_outputs)
        self.latest_experience = exp
        return exp

    def advance_training(self) -> List[Experience]:
        buffer: List[Experience] = [self.latest_experience]
        any_agent_done = False

        while not any_agent_done:
            exp = self.take_step()
            buffer.append(exp)
            for brain_name, brain_info in exp.brain_info.items():
                if any(brain_info.local_done):
                    any_agent_done = True
                    break
        return buffer


class ThreadedActorManager:
    def __init__(self, env_factory: Callable[[int], UnityEnvironment], n_env: int = 1):
        _am_futures = [self.create_actor_manager(env_factory, i) for i in range(n_env)]
        self._ams: List[ActorManager] = [f.result() for f in futures.wait(_am_futures)[0]]
        self._advance_futures: List[Optional[Future]] = [None] * len(self._ams)

    @threaded
    def create_actor_manager(self, env_factory, i):
        return ActorManager(env_factory, i)

    def get_external_brains(self):
        return self._ams[0].get_external_brains()

    def get_reset_parameters(self):
        return self._ams[0].get_external_brains()

    def set_policies(self, policies: Dict[str, Policy]):
        for i, am in enumerate(self._ams):
            if self._advance_futures[i]:
                self._advance_futures[i].cancel()
            am.set_policies(policies)

    def set_default_reset_params(self, reset_params: Dict[str, str]):
        for am in self._ams:
            am.set_default_reset_params(reset_params)

    # def load_policies(self, serialized_policies: Dict[str, PolicyDef]):
    #     self.policies = {}
    #     for brain_name, policy_def in serialized_policies.items():
    #         self.policies[brain_name] = Policy.load_from_policy_def(policy_def, self.get_external_brains()[brain_name])

    def reset(self, config=None) -> Experience:
        accumulated_brain_info = None
        for worker_id, am in enumerate(self._ams):
            experience = am.reset(config)
            all_brain_info = experience.brain_info
            for brain_name, brain_info in all_brain_info.items():
                for j in range(len(brain_info.agents)):
                    brain_info.agents[j] = str(worker_id) + '-' + str(brain_info.agents[j])

                if accumulated_brain_info:
                    accumulated_brain_info[brain_name].merge(brain_info)
            if not accumulated_brain_info:
                accumulated_brain_info = deepcopy(all_brain_info)
        return Experience(accumulated_brain_info, None)

    def is_global_done(self):
        dones = [am.is_global_done() for am in self._ams]
        return all(dones)

    def close(self):
        [am.close() for am in self._ams]

    @threaded
    def _advance_training(self, am: ActorManager, worker_id: int):
        return worker_id, am.advance_training()

    def advance_training(self) -> List[Experience]:
        for i, af in enumerate(self._advance_futures):
            if not af or af.done():
                self._advance_futures[i] = self._advance_training(self._ams[i], i)

        completed = list(futures.wait(self._advance_futures, return_when=futures.FIRST_COMPLETED)[0])[0]
        worker_id, experiences = completed.result()
        experiences = deepcopy(experiences)
        for i in range(len(experiences)):
            experiences[i] = deepcopy(experiences[i])

        for experience in experiences:
            for brain_name, brain_info in experience.brain_info.items():
                for i in range(len(brain_info.agents)):
                    brain_info.agents[i] = str(worker_id) + '-' + str(brain_info.agents[i])
        return experiences
