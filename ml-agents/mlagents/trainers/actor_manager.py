from typing import *

from mlagents.envs import AllBrainInfo, UnityEnvironment
from mlagents.trainers import PolicyDef, Policy
from mlagents.trainers.ppo import PPOPolicy


class Experience(NamedTuple):
    brain_info: AllBrainInfo
    all_action_outputs: Optional[Dict[str, Dict[str, Any]]]


class ActorManager:
    def __init__(self, env: Callable[[int], UnityEnvironment]):
        self.env = env(0)
        self.latest_experience = None
        self._default_reset_params = None
        self.policies = {}

    def get_external_brains(self):
        external_brains = {}
        for brain_name in self.env.external_brain_names:
            external_brains[brain_name] = self.env.brains[brain_name]
        return external_brains

    def get_reset_parameters(self):
        return self.env._resetParameters

    def set_policies(self, policies: Dict[str, Policy]):
        self.policies = policies

    def set_default_reset_params(self, reset_params: Dict[str, str]):
        self._default_reset_params = reset_params

    # def load_policies(self, serialized_policies: Dict[str, PolicyDef]):
    #     self.policies = {}
    #     for brain_name, policy_def in serialized_policies.items():
    #         if policy_def.type == 'PPOPolicy':
    #             policy = PPOPolicy(policy_def.seed, policy_def.brain, policy_def.trainer_parameters, True, False)
    #             policy.load_from_memory(policy_def.values)
    #             self.policies[brain_name] = policy

    def reset(self, config=None) -> AllBrainInfo:
        reset_params = {}
        if self._default_reset_params is not None and config is not None:
            reset_params = self._default_reset_params.copy()
            reset_params.update(config)
        elif self._default_reset_params is not None:
            reset_params = self._default_reset_params
        elif config is not None:
            reset_params = config

        if reset_params is not None:
            return self.env.reset(config=reset_params)
        else:
            return self.env.reset()

    def is_global_done(self):
        return self.env.global_done

    def close(self):
        self.env.close()

    def take_step(self):
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

    def advance_training(self):
        buffer: List[Experience] = [self.latest_experience]
        any_agent_done = False

        while not any_agent_done:
            self.latest_experience = self.take_step()
            buffer.append(self.latest_experience)
            for brain_name, brain_info in self.latest_experience.brain_info.items():
                if any(brain_info.local_done):
                    any_agent_done = True
                    break
        return buffer

