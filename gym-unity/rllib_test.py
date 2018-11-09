import argparse
import gym
import random

import ray
from ray import tune
from ray.rllib.agents.pg.pg_policy_graph import PGPolicyGraph
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune import run_experiments
from ray.tune.registry import register_env
import tensorflow as tf

from gym_unity.envs.unity_env import UnityEnv
from gym import spaces
import numpy as np
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.a3c as a3c

from tensorflow.python.tools import freeze_graph

if __name__ == "__main__":
    ray.init()

    volume_path = "/Users/jharper/Documents/repos/ml-agents/unity-volume/"
    environment_filename = volume_path + "Snoopy_Linux"
    register_env("unity_env", lambda config: UnityEnv(config, environment_filename, config.worker_index, True))

    # Set up PPO
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_workers"] = 0
    config["num_gpus"] = 0
    config["num_gpus_per_worker"] = 0
    config["num_cpus_per_worker"] = 0
    config["sample_batch_size"] = 8
    config["train_batch_size"] = 4
    config["sgd_minibatch_size"] = 2

    agent = ppo.PPOAgent(config=config, env="unity_env")
    for i in range(1):
        result = agent.train()
        #print(pretty_print(result))

    graph = agent.local_evaluator.policy_map['default'].sess.graph
    tf.train.write_graph(graph, volume_path, 'raw_graph_def.pb', as_text=False)
    saver = tf.train.Saver(agent.local_evaluator.policy_map['default']._variables)
    last_checkpoint = volume_path + '/latest.cptk'
    sess = agent.local_evaluator.policy_map['default'].sess
    saver.save(sess, last_checkpoint)
    ckpt = tf.train.get_checkpoint_state(volume_path)
    graph_def = graph.as_graph_def()

    freeze_graph.freeze_graph(
        input_graph=self.volume_path + '/raw_graph_def.pb',
        input_binary=True,
        input_checkpoint=ckpt.model_checkpoint_path,
        output_node_names=target_nodes,
        output_graph=(self.model_path + '/' + self.env_name + '_'
                      + self.run_id + '.bytes'),
        clear_devices=True, initializer_nodes='', input_saver='',
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0'
    )