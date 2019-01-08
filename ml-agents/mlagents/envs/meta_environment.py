import ray

from .environment import *
from .brain import *


@ray.remote
class Exec(object):
    def __init__(self, file_name=None, worker_id=0,
                 base_port=5005, curriculum=None,
                 seed=0, docker_training=False):
        self.env = UnityEnvironment(file_name, worker_id, base_port, curriculum, seed, docker_training)
        # self.env.reset()

    def reset(self, config=None, train_mode=True):
        return self.env.reset(config, train_mode)

    def step(self, vector_action=None, memory=None, text_action=None):
        return self.env.step(vector_action, memory, text_action)

    def close(self):
        self.env.close()

    def logfile_path(self):
        return self.env.logfile_path

    def brains(self):
        return self.env.brains

    def global_done(self):
        return self.env.global_done

    def academy_name(self):
        return self.env.academy_name

    def number_brains(self):
        return self.env.number_brains

    def number_external_brains(self):
        return self.env.number_external_brains

    def brain_names(self):
        return self.env.brain_names

    def external_brain_names(self):
        return self.env.external_brain_names


class MetaUnityEnvironment(object):
    def __init__(self, file_name=None, worker_id=0,
                 base_port=5005, curriculum=None,
                 seed=0, docker_training=False,
                 num_env=1):
        self.n_agents = None

        ray.init()
        self.actors = [Exec.remote(file_name, worker_id + i, base_port, curriculum, seed, docker_training)
                       for i in range(num_env)]
        # TODO : Multibrain
        self.single_brain_name = self.external_brain_names[0]

    def reset(self, train_mode=True, config=None, lesson=None):
        results = ray.get([c.reset.remote(config, train_mode) for c in self.actors])
        return self._unify_brain_info(results)

    def step(self, vector_action=None, memory=None, text_action=None, value = None):
        print("Calling step...")
        if vector_action!= None :
            tmp = self._split_input(vector_action, memory, text_action)
            results = ray.get([c.step.remote(tmp[i]) for i, c in enumerate(self.actors)])
        else:
            results = ray.get([c.step.remote(None) for i, c in enumerate(self.actors)])
        print("Calling step...")
        return self._unify_brain_info(results)

    def close(self):
        ray.get([c.close.remote() for c in self.actors])

    def _split_input(self, vector_action_dict, memory, text_action):
        # TODO : Memories and text actions
        vector_action = vector_action_dict[self.single_brain_name]
        start_index = 0
        vector_actions = []
        for t in self.n_agents:
            vector_actions +=[vector_action[start_index:start_index+t, :]]
            start_index += t
        return vector_actions

    def _unify_brain_info(self, brain_info_list_dict):
        brain_info_list = [x[self.single_brain_name] for x in brain_info_list_dict]
        self.n_agents = [len(x.agents) for x in brain_info_list]
        # visual_observation=[]
        text_observations = []
        rewards = []
        local_done = []
        max_reached = []
        agents = []
        previous_text_actions = []
        for x, b_i in enumerate(brain_info_list):
            # TODO : Multiple visual observations
            # visual_observation.extend(b_i.visual_observations)
            text_observations.extend(b_i.text_observations)
            rewards.extend(b_i.rewards)
            local_done.extend(b_i.local_done)
            max_reached.extend(b_i.max_reached)
            agents.extend([x*100000000 + y for y in b_i.agents])
            previous_text_actions.extend(b_i.previous_text_actions)
        if len(brain_info_list[0].visual_observations) > 0:
            visual_observations = [np.concatenate([x.visual_observations[0] for x in brain_info_list])]
        else:
            visual_observations = []
        vector_observations = np.concatenate([x.vector_observations for x in brain_info_list])
        memories = np.concatenate([x.memories for x in brain_info_list])
        previous_vector_actions = np.concatenate([x.previous_vector_actions for x in brain_info_list])
        action_mask = np.concatenate([x.action_masks for x in brain_info_list])
        return {self.single_brain_name:
                    BrainInfo(
                        visual_observations, vector_observations, text_observations, memories,
                        reward=rewards, agents=agents, local_done=local_done,
                        vector_action=previous_vector_actions, text_action=previous_text_actions,
                        max_reached=max_reached, action_mask=action_mask)}

    @property
    def logfile_path(self):
        return ray.get(self.actors[0].logfile_path.remote())

    @property
    def brains(self):
        return ray.get(self.actors[0].brains.remote())

    @property
    def global_done(self):
        return ray.get(self.actors[0].global_done.remote())

    @property
    def academy_name(self):
        return ray.get(self.actors[0].academy_name.remote())

    @property
    def number_brains(self):
        return ray.get(self.actors[0].number_brains.remote())

    @property
    def number_external_brains(self):
        return ray.get(self.actors[0].number_external_brains.remote())

    @property
    def brain_names(self):
        return ray.get(self.actors[0].brain_names.remote())

    @property
    def external_brain_names(self):
        return ray.get(self.actors[0].external_brain_names.remote())
