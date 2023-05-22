import argparse
import logging
from pathlib import Path
import sys

from hulc2.evaluation.evaluation import Evaluation

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
from pytorch_lightning import seed_everything
logger = logging.getLogger(__name__)



def main():
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset root directory.")

    # arguments for loading default model
    parser.add_argument(
        "--train_folder", type=str, help="If calvin_agent was used to train, specify path to the log dir."
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Comma separated list of epochs for which checkpoints will be loaded",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path of the checkpoint",
    )
    parser.add_argument(
        "--last_k_checkpoints",
        type=int,
        help="Specify the number of checkpoints you want to evaluate (starting from last). Only used for calvin_agent.",
    )

    # arguments for loading custom model or custom language embeddings
    parser.add_argument(
        "--custom_model", action="store_true", help="Use this option to evaluate a custom model architecture."
    )
    parser.add_argument("--custom_lang_embeddings", action="store_true", help="Use custom language embeddings.")
    parser.add_argument("--cameras", type=str)
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.")

    parser.add_argument("--log_dir", default=None, type=str, help="Where to log the evaluation results.")

    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    parser.add_argument("--n_completed", default=0, type=int, help="Save rollout after n_completed succesful tasks")

    parser.add_argument("--aff_train_folder", default=None, help="Affordance model train folder to use model-based model-free combination method.")
    parser.add_argument("--aff_checkpoint", default="val_err", help="Affordance model checkpoint name w/extension to use model-based model-free combination method.")
    # Overriding scene
    parser.add_argument(
        "--scene", default=None, type=str, help="Name of scene file inside /config/scene/ without extension"
    )

    args = parser.parse_args()

    # Do not change
    args.ep_len = 360
    args.num_sequences = 1000

    assert "train_folder" in args

    if args.checkpoint:
        checkpoints = [Path("epoch=%s.ckpt" % args.checkpoint)]
    else:
        checkpoints = [Path("epoch=%s.ckpt" % int(chk)) for chk in args.checkpoints.split(',')]

    results = {}
    plans = {}
    env = None
    for checkpoint in checkpoints:
        eval = Evaluation(args, checkpoint, env)
        env = eval.env
        results[checkpoint], plans[checkpoint] = eval.evaluate_policy(args)
    eval.print_and_save(results, plans, args)


if __name__ == "__main__":
    main()
