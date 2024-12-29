#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""Implements evaluation of ``habitat.Agent`` inside ``habitat.Env``.
``habitat.Benchmark`` creates a ``habitat.Env`` which is specified through
the ``config_env`` parameter in constructor. The evaluation is task agnostic
and is implemented through metrics defined for ``habitat.EmbodiedTask``.
"""

import os
from collections import defaultdict
from typing import Dict, Optional
from habitat.config.default import get_config
from habitat.core.agent import Agent
from habitat.core.env import Env
from habitat.core.logging import logger
import habitat_sim.agent
import json
import imageio
from pathlib import Path
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision.utils import save_image

class Benchmark:
    r"""Benchmark for evaluating agents in environments."""

    def __init__(
        self, config_paths: Optional[str] = None, eval_remote: bool = False, split_l: int = -1, split_r: int = -1
    ) -> None:
        r"""..

        :param config_paths: file to be used for creating the environment
        :param eval_remote: boolean indicating whether evaluation should be run remotely or locally
        """
        config_env = get_config(config_paths)
        config_env.DATASET.defrost()
        config_env.DATASET.SPLIT_L = split_l
        config_env.DATASET.SPLIT_R = split_r
        config_env.DATASET.freeze()
        self._eval_remote = eval_remote

        if self._eval_remote is True:
            self._env = None
        else:
            self._env = Env(config=config_env)

    def remote_evaluate(
        self, agent: "Agent", num_episodes: Optional[int] = None
    ):
        # The modules imported below are specific to habitat-challenge remote evaluation.
        # These modules are not part of the habitat-lab repository.
        import pickle
        import time

        import evalai_environment_habitat  # noqa: F401
        import evaluation_pb2
        import evaluation_pb2_grpc
        import grpc

        def pack_for_grpc(entity):
            return pickle.dumps(entity)

        def unpack_for_grpc(entity):
            return pickle.loads(entity)

        def remote_ep_over(stub):
            res_env = unpack_for_grpc(
                stub.episode_over(evaluation_pb2.Package()).SerializedEntity
            )
            return res_env["episode_over"]

        env_address_port = os.environ.get("EVALENV_ADDPORT", "localhost:8085")
        channel = grpc.insecure_channel(env_address_port)
        stub = evaluation_pb2_grpc.EnvironmentStub(channel)

        base_num_episodes = unpack_for_grpc(
            stub.num_episodes(evaluation_pb2.Package()).SerializedEntity
        )
        num_episodes = base_num_episodes["num_episodes"]

        agg_metrics: Dict = defaultdict(float)

        count_episodes = 0

        while count_episodes < num_episodes:
            agent.reset()
            res_env = unpack_for_grpc(
                stub.reset(evaluation_pb2.Package()).SerializedEntity
            )

            while not remote_ep_over(stub):
                obs = res_env["observations"]
                action = agent.act(obs)

                res_env = unpack_for_grpc(
                    stub.act_on_environment(
                        evaluation_pb2.Package(
                            SerializedEntity=pack_for_grpc(action)
                        )
                    ).SerializedEntity
                )

            metrics = unpack_for_grpc(
                stub.get_metrics(
                    evaluation_pb2.Package(
                        SerializedEntity=pack_for_grpc(action)
                    )
                ).SerializedEntity
            )

            for m, v in metrics["metrics"].items():
                agg_metrics[m] += v
            count_episodes += 1

        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

        stub.evalai_update_submission(evaluation_pb2.Package())

        return avg_metrics

    def local_evaluate(
        self, agent: "Agent", num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        if num_episodes is None:
            num_episodes = len(self._env.episodes)
        else:
            assert num_episodes <= len(self._env.episodes), (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(
                    num_episodes, len(self._env.episodes)
                )
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"
        print(num_episodes)
        agg_metrics: Dict = defaultdict(float)

        count_episodes = 0
        all_metrics = []
        all_metrics_0 = []
        all_metrics_avg = []
        count_success = 0
        agent.benchmark = self
        while count_episodes < num_episodes:
            observations = self._env.reset()
            agent.reset(obj_goal=self._env.current_episode.object_category)
            metrics = self._env.get_metrics()
            all_metrics_0.append(metrics)
            while not self._env.episode_over:
                action = agent.act(observations)
                if agent.total_steps == 500:
                    metrics = self._env.get_metrics()
                observations = self._env.step(action)
                agent.update_metrics(self._env.get_metrics())

            metrics = self._env.get_metrics()
            all_metrics.append(metrics)
            print(count_episodes, metrics)
            if metrics['success'] == 1:
                count_success += 1
            for m, v in metrics.items():
                if isinstance(v, dict):
                    for sub_m, sub_v in v.items():
                        agg_metrics[m + "/" + str(sub_m)] += sub_v
                else:
                    agg_metrics[m] += v
            count_episodes += 1
            avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}
            all_metrics_avg.append(avg_metrics)
            for k,v in avg_metrics.items():
                logger.info("{}: {}".format(k, v))
            
            experiment_name = agent.experiment_name

            path = f'data/results/{experiment_name}'
            if not os.path.exists(path):
                os.makedirs(path)

            with open(f'data/results/{experiment_name}/results.txt', 'w') as fp:
                for item in all_metrics:
                    # write each item on a new line
                    fp.write("%s\n" % item)
                    
            with open(f'data/results/{experiment_name}/results_0.txt', 'w') as fp:
                for item in all_metrics_0:
                    # write each item on a new line
                    fp.write("%s\n" % item)

            with open(f'data/results/{experiment_name}/results_avg.txt', 'w') as fp:
                for item in all_metrics_avg:
                    # write each item on a new line
                    fp.write("%s\n" % item)

            def result_text_list():
                success_text = 'success' if metrics['success'] == 1 else 'fail'
                text_list = [
                    success_text,
                    '',
                    'distance_to_goal_mean: {}'.format(avg_metrics['distance_to_goal']),
                    'success_rate: {}'.format(avg_metrics['success']),
                    'spl_mean: {}'.format(avg_metrics['spl']),
                    'softspl_mean: {}'.format(avg_metrics['softspl']),
                ]
                return text_list
            
            def add_text(image: torch.Tensor, text_list: list, coordinate=(5, 5)):
                image = image.numpy()
                image = Image.fromarray(image)
                draw = ImageDraw.Draw(image)
                text = ''
                for t in text_list:
                    text = text + t + '\n'
                draw.text(coordinate, text, fill='black')
                image = np.array(image)
                return image

            def save_video(experiment_name, metrics):
                image_dir = f"figures/{experiment_name}/image"
                image_file_list = os.listdir(image_dir)
                image_file_list = sorted(image_file_list)
                image_path_list = [os.path.join(image_dir, image_file) for image_file in image_file_list]
                save_video_dir = f'figures/{experiment_name}/video'
                episode_index = count_episodes - 1
                success = 'success' if metrics['success'] == 1 else 'fail'
                save_video_path = f'{save_video_dir}/vid_{episode_index:06d}_{success}.mp4'
                if not os.path.exists(save_video_dir):
                    os.makedirs(save_video_dir)
                frame = cv2.imread(image_path_list[0])  
                height, width, layers = frame.shape  
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(save_video_path, fourcc, 4.0, (width, height))
                for image in image_path_list:  
                    video.write(cv2.imread(image))
                text_list = result_text_list()
                image = torch.tensor(cv2.imread(image))
                image = add_text(image, text_list, (5, 350))
                video.release()
            
            # save_video(experiment_name, metrics)  # visulization
            
            
        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

        return avg_metrics

    def evaluate(
        self, agent: "Agent", num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        r"""..

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the
            evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """

        if self._eval_remote is True:
            return self.remote_evaluate(agent, num_episodes)
        else:
            return self.local_evaluate(agent, num_episodes)
        
    def visualization(self, agent, action, observations):
        if not hasattr(agent, 'agent_state_image') or agent.demo_img is None:
            return
        save_image_dir = f'figures/{agent.experiment_name}/image'
        if not os.path.exists(save_image_dir):
            os.makedirs(save_image_dir)
        save_image(agent.demo_img, os.path.join(save_image_dir, f'img_{agent.navigate_steps:06d}.png'))
        agent.demo_img = None
