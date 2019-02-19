from typing import *

from mlagents.envs import UnityEnvironment
from mlagents.trainers import PolicyDef, ActorManager, Experience
from multiprocessing import Process, Queue, Manager

class ActorMessage(NamedTuple):
    name: str
    payload: Any = None


class ActorSubprocessMessage(NamedTuple):
    name: str
    worker_id: int
    payload: Any = None


def run_subprocess_actor_manager(
    outbox: Queue, inbox: Queue, env_factory: Callable[[int], UnityEnvironment], worker_id: int
):
    am = ActorManager(env_factory, worker_id)
    policy_count = 0
    done = False
    try:
        while not done:
            msg: ActorMessage = inbox.get()
            if not msg:
                continue
            if msg.name == 'get_external_brains':
                outbox.put(
                    ActorSubprocessMessage('get_external_brains', worker_id, am.get_external_brains())
                )
            elif msg.name == 'get_reset_parameters':
                outbox.put(
                    ActorSubprocessMessage('get_reset_parameters', am.get_reset_parameters())
                )
            elif msg.name == 'set_default_reset_params':
                am.set_default_reset_params(msg.payload)
            elif msg.name == 'load_policies':
                am.load_policies(msg.payload)
                policy_count += 1
            elif msg.name == 'reset':
                outbox.put(
                    ActorSubprocessMessage('reset', worker_id, am.reset(msg.payload))
                )
            elif msg.name == 'is_global_done':
                outbox.put(
                    ActorSubprocessMessage('is_global_done', am.is_global_done())
                )
            elif msg.name == 'advance_training':
                outbox.put(
                    ActorSubprocessMessage('advance_training', worker_id, (policy_count, am.advance_training()))
                )
            elif msg.name == 'close':
                am.close()
                inbox.empty()
                done = True
    except KeyboardInterrupt:
        print('ActorManager subprocess ' + str(worker_id) + ': got KeyboardInterrupt')
    finally:
        am.close()


class SubprocessActorManagerState(NamedTuple):
    process: Process
    worker_id: int
    outbox: Queue


class SubprocessActorManager:
    def __init__(self, env_factory: Callable[[int], UnityEnvironment], n_env: int = 1):
        self.envs = []
        manager = Manager()
        self.inbox = manager.Queue()
        for i in range(n_env):
            outbox = manager.Queue()
            p = Process(target=run_subprocess_actor_manager, args=(self.inbox, outbox, env_factory, i))
            p.start()
            self.envs.append(SubprocessActorManagerState(p, i, outbox))
        self._external_brains = None
        self._reset_params = None
        self._not_yet_advanced = True
        self.policy_count = 0

    def get_external_brains(self):
        return self._get_from_worker('get_external_brains', 0)

    def get_reset_parameters(self):
        return self._get_from_worker('get_reset_parameters', 0)

    def set_default_reset_params(self, reset_params: Dict[str, str]):
        self._broadcast_message(ActorMessage('set_default_reset_params', reset_params))

    def load_policies(self, serialized_policies: Dict[str, PolicyDef]):
        # if self.policy_count == 0:
        self._broadcast_message(ActorMessage('load_policies', serialized_policies))
        self.policy_count += 1

    def reset(self, config=None) -> Experience:
        self._broadcast_message(ActorMessage('reset', config))
        reset_results: List[ActorSubprocessMessage] = []
        while len(reset_results) < len(self.envs):
            next_reset = self._block_until_message('reset')
            reset_results.append(next_reset)
        accumulated_brain_info = None
        for reset_result in reset_results:
            all_brain_info = reset_result.payload.brain_info
            for brain_name, brain_info in all_brain_info.items():
                for i in range(len(brain_info.agents)):
                    brain_info.agents[i] = 'worker' + str(reset_result.worker_id) + str(brain_info.agents[i])

                if accumulated_brain_info:
                    accumulated_brain_info[brain_name].merge(brain_info)
            if not accumulated_brain_info:
                accumulated_brain_info = all_brain_info
        return Experience(accumulated_brain_info, None)

    def is_global_done(self):
        return False

    def close(self):
        for env in self.envs:
            env.process.join()

    def advance_training(self) -> List[Experience]:
        for env in self.envs:
            if env.outbox.empty():
                env.outbox.put(ActorMessage('advance_training'))

        next_advance: ActorSubprocessMessage = self._block_until_message('advance_training')
        policy_num = next_advance.payload[0]
        while policy_num != self.policy_count:
            next_advance = self._block_until_message('advance_training')
            policy_num = next_advance.payload[0]

        experiences = next_advance.payload[1]
        for experience in experiences:
            for brain_name, brain_info in experience.brain_info.items():
                for i in range(len(brain_info.agents)):
                    brain_info.agents[i] = 'worker' + str(next_advance.worker_id) + str(brain_info.agents[i])
        return experiences

    def _broadcast_message(self, msg: ActorMessage):
        for env in self.envs:
            env.outbox.put(msg)

    def _get_from_worker(self, name: str, worker_id: int):
        self._outbox(worker_id).put(ActorMessage(name))
        response = self._block_until_message(name)
        return response.payload

    def _block_until_message(self, name: str):
        next_message = self.inbox.get()
        while next_message and next_message.name != name:
            self.inbox.put(next_message)
            next_message = self.inbox.get()
        return next_message

    def _outbox(self, i: int):
        return self.envs[i].outbox
