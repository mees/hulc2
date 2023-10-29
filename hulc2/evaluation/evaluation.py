from collections import Counter, defaultdict
import json
import logging
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from tqdm.auto import tqdm

from hulc2.evaluation.manager_aff_lmp import PolicyManager as AffLMPManager
from hulc2.evaluation.multistep_sequences import get_sequences
from hulc2.evaluation.utils import get_env_state_for_initial_condition

logger = logging.getLogger(__name__)


class Evaluation:
    def __init__(self, args, checkpoint, env=None) -> None:
        self.use_affordances = args.aff_train_folder is not None
        self.policy_manager = AffLMPManager(
            train_folder=args.aff_train_folder,
            checkpoint=args.aff_checkpoint,
            debug=args.debug,
            save_viz=args.save_viz,
            use_affordances=self.use_affordances,
        )
        self.save_viz = args.save_viz
        self.n_completed = args.n_completed
        scene = args.scene
        if args.scene is not None:
            s = "config/scene/%s.yaml" % args.scene
            scene = OmegaConf.load(Path(__file__).parents[2] / s)

        camera_conf = args.cameras
        if args.cameras is not None:
            s = "config/cameras/%s.yaml" % args.cameras
            camera_conf = OmegaConf.load(Path(__file__).parents[2] / s)

        model, env, _, lang_embeddings = self.policy_manager.get_default_model_and_env(
            args.train_folder,
            args.dataset_path,
            checkpoint,
            device_id=args.device,
            scene=scene,
            env=env,
            camera_conf=camera_conf,
        )
        self.model = model
        self.env = env
        self.lang_embeddings = lang_embeddings

    def get_log_dir(self, log_dir):
        if log_dir is not None:
            log_dir = Path(log_dir)
            os.makedirs(log_dir, exist_ok=True)
        else:
            log_dir = Path(__file__).parents[3] / "evaluation"
            if not log_dir.exists():
                log_dir = Path("/tmp/evaluation")
        print(f"logging to {log_dir}")
        return log_dir

    def count_success(self, results):
        count = Counter(results)
        step_success = []
        for i in range(1, 6):
            n_success = sum(count[j] for j in reversed(range(i, 6)))
            sr = n_success / len(results)
            step_success.append(sr)
        return step_success

    def print_and_save(self, total_results, plan_dicts, args):
        log_dir = self.get_log_dir(args.log_dir)

        sequences = get_sequences(args.num_sequences)

        current_data = {}
        ranking = {}
        for checkpoint, results in total_results.items():
            epoch = checkpoint.stem.split("=")[1]
            print(f"Results for Epoch {epoch}:")
            avg_seq_len = np.mean(results)
            ranking[epoch] = avg_seq_len
            chain_sr = {i + 1: sr for i, sr in enumerate(self.count_success(results))}
            print(f"Average successful sequence length: {avg_seq_len}")
            print("Success rates for i instructions in a row:")
            for i, sr in chain_sr.items():
                print(f"{i}: {sr * 100:.1f}%")

            cnt_success = Counter()
            cnt_fail = Counter()

            for result, (_, sequence) in zip(results, sequences):
                for successful_tasks in sequence[:result]:
                    cnt_success[successful_tasks] += 1
                if result < len(sequence):
                    failed_task = sequence[result]
                    cnt_fail[failed_task] += 1

            total = cnt_success + cnt_fail
            task_info = {}
            for task in total:
                task_info[task] = {"success": cnt_success[task], "total": total[task]}
                print(
                    f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%"
                )

            data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}

            current_data[epoch] = data

            print()
        previous_data = {}
        try:
            with open(log_dir / "results.json", "r") as file:
                previous_data = json.load(file)
        except FileNotFoundError:
            pass

        json_data = {**previous_data, **current_data}
        best_model = max(ranking, key=ranking.get)
        best_model_data = {"epoch": best_model, **json_data[best_model]}
        json_data["best"] = best_model_data
        with open(log_dir / "results.json", "w") as file:
            json.dump(json_data, file)
        print(f"Best model: epoch {best_model} with average sequences length of {max(ranking.values())}")

        for checkpoint, plan_dict in plan_dicts.items():
            epoch = checkpoint.stem.split("=")[1]

            ids, labels, plans, latent_goals = zip(
                *[
                    (i, label, latent_goal, plan)
                    for i, (label, plan_list) in enumerate(plan_dict.items())
                    for latent_goal, plan in plan_list
                ]
            )
            latent_goals = torch.cat(latent_goals)
            plans = torch.cat(plans)
            np.savez(
                f"{log_dir / f'tsne_data_{epoch}.npz'}", ids=ids, labels=labels, plans=plans, latent_goals=latent_goals
            )

    def evaluate_policy(self, args):
        conf_dir = Path(__file__).absolute().parents[2] / "config"
        task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
        task_oracle = hydra.utils.instantiate(task_cfg)
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

        eval_sequences = get_sequences(args.num_sequences)
        # eval_sequences = [eval_sequences[i] for i in [5,12,24,30,33,34,51,52,53,61,77,80]]
        # eval_sequences = eval_sequences[:250]
        results = []
        plans = defaultdict(list)

        if not args.debug:
            eval_sequences = tqdm(eval_sequences, position=0, leave=True)

        saved_sequences = []
        for i, (initial_state, eval_sequence) in enumerate(eval_sequences):
            seq_annotations = [val_annotations[subtask][0] for subtask in eval_sequence]

            # The model uses the sequence counter to generate the image filenames
            self.model.save_dir["sequence_counter"] = i
            result = self.evaluate_sequence(task_oracle, initial_state, eval_sequence, val_annotations, args, plans)
            results.append(result)

            if (result <= 3 and not self.use_affordances) or (
                result >= 4 and self.use_affordances
            ):  # self.n_completed:
                saved_sequences.append(i)
                if self.save_viz:
                    self.model.save_sequence_txt("sequence_%04d" % i, seq_annotations)
                    self.model.save_sequence()

            self.model.save_dir["rollout_counter"] = 0
            self.model.sequence_data = {}
            if not args.debug:
                eval_sequences.set_description(
                    " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(self.count_success(results))]) + "|"
                )

        self.model.save_sequence_txt("sequence_ids", saved_sequences)
        return results, plans

    def evaluate_sequence(self, task_checker, initial_state, eval_sequence, val_annotations, args, plans):
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)

        # Always start robot from neutral position ?
        self.env.reset(robot_obs=None, scene_obs=scene_obs)

        success_counter = 0
        if args.debug:
            time.sleep(1)
            print()
            print()
            print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
            print("Subtask: ", end="")
        for subtask in eval_sequence:
            success = self.policy_manager.rollout(
                self.env, self.model, task_checker, args, subtask, self.lang_embeddings, val_annotations, plans
            )
            if success:
                success_counter += 1
                self.model.save_dir["rollout_counter"] += 1
            else:
                return success_counter
        return success_counter
