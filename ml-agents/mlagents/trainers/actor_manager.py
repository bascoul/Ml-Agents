from typing import *

from mlagents.envs import BrainInfo, AllBrainInfo, UnityEnvironment
from mlagents.trainers import PolicyDef, Policy


class Experience(NamedTuple):
    brain_info: AllBrainInfo
    all_action_outputs: Optional[Dict[str, Dict[str, Any]]]


# class ActorManagerBuffer:
#     def __init__(self, initial_experience: BrainInfo):
#         self.agents = initial_experience.agents
#         self.agent_buffers = {}
#

class ActorManager:
    def __init__(self, env_factory: Callable[[int], UnityEnvironment], worker_id: int=0):
        self.env: UnityEnvironment = env_factory(worker_id)
        self.latest_experience = None
        self._default_reset_params = None
        self.policies = {}

    def get_external_brains(self):
        return self.env.external_brains

    def get_reset_parameters(self):
        return self.env.reset_parameters

    def set_policies(self, policies: Dict[str, Policy]):
        self.policies = policies

    def set_default_reset_params(self, reset_params: Dict[str, str]):
        self._default_reset_params = reset_params

    def load_policies(self, serialized_policies: Dict[str, PolicyDef]):
        self.policies = {}
        for brain_name, policy_def in serialized_policies.items():
            if brain_name in self.policies:
                self.policies[brain_name].load_from_memory(serialized_policies[brain_name].values)
            else:
                self.policies[brain_name] = Policy.load_from_policy_def(
                    policy_def, self.get_external_brains()[brain_name]
                )

    def reset(self, config=None) -> Experience:
        reset_params = {}
        if self._default_reset_params is not None and config is not None:
            reset_params = self._default_reset_params.copy()
            reset_params.update(config)
        elif self._default_reset_params is not None:
            reset_params = self._default_reset_params
        elif config is not None:
            reset_params = config

        latest_brain_infos = \
            self.env.reset(config=reset_params) if reset_params else \
            self.env.reset()

        self.latest_experience = Experience(
            latest_brain_infos, None
        )
        return self.latest_experience

    def is_global_done(self):
        return self.env.global_done

    def close(self):
        self.env.close()

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
        return Experience(
            self.env.step(
                vector_action=take_action_vector,
                memory=take_action_memories,
                text_action=take_action_text,
                value=take_action_value
            ),
            take_action_outputs
        )

    def advance_training(self) -> List[Experience]:
        buffer: List[Experience] = [self.latest_experience]

        while True:
            self.latest_experience = self.take_step()
            buffer.append(self.latest_experience)
            for brain_name, brain_info in self.latest_experience.brain_info.items():
                if any(brain_info.local_done):
                    break
            if len(buffer) >= 128:
                break
        return buffer

