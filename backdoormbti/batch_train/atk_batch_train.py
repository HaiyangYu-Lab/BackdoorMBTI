"""
This file implements the entire process of a backdoor attack and serves as the main entry point for the BackdoorMBTI backdoor attack.

The basic structure of this file is as follows:
1. Basic Setup: Parameters, logging, etc.
2. Data Loading and Poisoning
3. Model Loading
4. Training, Evaluation, and Saving
"""

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.append("../")
import logging

from batch_train.random_params import benign_random_params
from configs.settings import BASE_DIR
from eval.sl_learning_eval import SupervisedLearningEval
from train.sl_learning_train import SupervisedLearningTrain
from utils.args import add_yaml_to_args, init_args
from utils.data import BadSet, get_dataloader, load_dataset
from utils.io import (
    get_cfg_path_by_args,
    get_log_path_by_args,
    get_poison_ds_path_by_args,
    init_folders,
)
from utils.log import configure_logger
from utils.model import load_model, load_poisoned_model
from utils.wrapper import get_attack_by_args, get_data_spec_class_by_args


def _use_fast_host_to_device_path(args):
    return torch.cuda.is_available() and str(args.device).startswith("cuda")


IMAGE_ATTACK_RANDOMIZERS = {
    "blend": {
        "attack_train_blended_alpha": lambda rng, ref: ref["alpha"],
    },
    "adaptive_blend": {
        "adaptive_blend_alpha": lambda rng, ref: ref["alpha"],
        "adaptive_cover_rate": lambda rng, ref: round(float(rng.uniform(0.05, 0.5)), 3),
        "adaptive_blend_mask_rate": lambda rng, ref: round(float(rng.uniform(0.1, 0.8)), 3),
    },
    "bpp": {
        "random_crop": lambda rng, ref: int(rng.integers(0, 11)),
        "random_rotation": lambda rng, ref: int(rng.integers(0, 181)),
        "squeeze_num": lambda rng, ref: int(rng.integers(2, 65)),
        "dithering": lambda rng, ref: bool(rng.integers(0, 2)),
    },
    "sig": {
        "poison_type": lambda rng, ref: str(rng.choice(["sin", "ramp", "triangle"])),
    },
    "wanet": {
        "s": lambda rng, ref: round(float(rng.uniform(0, 1)), 6),
        "k": lambda rng, ref: int(rng.integers(1, ref["max_size"] + 1)),
        "cross_ratio": lambda rng, ref: round(float(rng.uniform(0, 0.5)), 3),
    },
    "refool": {
        "ghost_rate": lambda rng, ref: round(float(rng.uniform(0, 1)), 3),
        "alpha_b": lambda rng, ref: [round(float(v), 3) for v in rng.uniform(0.05, 0.5, 2)],
        "ghost_alpha": lambda rng, ref: round(float(rng.uniform(0.05, 0.5)), 3),
        "sigma": lambda rng, ref: int(rng.integers(1, 6)),
    },
}

TRIGGER_PATH_KEYS = (
    "patch_mask_path",
    "trigger_path",
    "attack_trigger_img_path",
    "ref_img_floder",
    "attack_train_replace_imgs_path",
    "attack_test_replace_imgs_path",
)


def _get_batch_seed(args):
    return int(getattr(args, "random_seed", 0) or 0) + int(getattr(args, "i", 0) or 0)


def _mmt_reference_setting(args, rng):
    max_size = int(min(getattr(args, "input_height", 28), getattr(args, "input_width", 28)))
    patch_sizes = [size for size in [2, 3, 4, 5, max_size] if 1 <= size <= max_size]
    p_size = int(rng.choice(patch_sizes))
    if p_size < max_size:
        alpha = float(rng.uniform(0.2, 0.6))
        if alpha > 0.5:
            alpha = 1.0
        loc = (
            int(rng.integers(0, max_size - p_size + 1)),
            int(rng.integers(0, max_size - p_size + 1)),
        )
    else:
        alpha = float(rng.uniform(0.05, 0.2))
        loc = (0, 0)

    pattern_num = int(rng.integers(1, p_size**2))
    pattern = np.zeros((p_size**2), dtype=int)
    pattern[rng.choice(np.arange(p_size**2), pattern_num, replace=False)] = 1
    pattern = pattern.reshape((p_size, p_size))
    return {
        "max_size": max_size,
        "p_size": p_size,
        "pattern": pattern.tolist(),
        "loc": loc,
        "alpha": round(alpha, 6),
        "target_y": int(rng.integers(0, int(args.num_classes))),
        "inject_p": round(float(rng.uniform(0.05, 0.5)), 3),
    }


def randomize_image_attack_params(args):
    if args.data_type != "image" or args.train_benign:
        return {}

    rng = np.random.default_rng(_get_batch_seed(args))
    ref = _mmt_reference_setting(args, rng)
    args.attack_target = ref["target_y"]
    args.pratio = ref["inject_p"]
    randomized = {"attack_target": args.attack_target, "pratio": args.pratio}

    for key, randomizer in IMAGE_ATTACK_RANDOMIZERS.get(args.attack_name, {}).items():
        if hasattr(args, key):
            value = randomizer(rng, ref)
            setattr(args, key, value)
            randomized[key] = value

    return {"seed": _get_batch_seed(args), "mmt_reference": ref, "randomized_params": randomized}


def make_image_batch_poison_artifacts(args, Attack, clean_train_set, clean_test_set):
    save_folder_path = _get_save_folder_path(args)
    save_folder_path.mkdir(parents=True, exist_ok=True)
    poison_dir = save_folder_path / f"{args.i}_poison_data"
    poison_dir.mkdir(parents=True, exist_ok=True)

    train_set_wrapper = Attack(clean_train_set, args, mode="train")
    train_set_wrapper.make_and_save_dataset(save_dir=poison_dir)
    test_set_wrapper = Attack(clean_test_set, args, mode="test", pop=False)
    test_set_wrapper.make_and_save_dataset(save_dir=poison_dir)

    train_path = poison_dir / f"{args.data_type}_{args.attack_name}_poison_train_set.pt"
    test_path = poison_dir / f"{args.data_type}_{args.attack_name}_poison_test_set.pt"
    model_train_path = save_folder_path / f"{args.i}_poison_train.pt"
    model_test_path = save_folder_path / f"{args.i}_poison_test.pt"
    shutil.copyfile(train_path, model_train_path)
    shutil.copyfile(test_path, model_test_path)
    return poison_dir, model_train_path, model_test_path


def _json_safe(value):
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_batch_attack_metadata(
    args,
    save_folder_path,
    random_metadata=None,
    poison_train_path=None,
    poison_test_path=None,
):
    save_folder_path = Path(save_folder_path)
    trigger_paths = {
        key: _json_safe(getattr(args, key)) for key in TRIGGER_PATH_KEYS if hasattr(args, key)
    }
    payload = {
        "model_index": int(args.i),
        "data_type": args.data_type,
        "dataset": args.dataset,
        "attack_name": args.attack_name,
        "model_name": args.model_name,
        "random_seed": getattr(args, "random_seed", None),
        "batch_seed": _get_batch_seed(args),
        "attack_target": getattr(args, "attack_target", None),
        "pratio": getattr(args, "pratio", None),
        "trigger_paths": trigger_paths,
        "randomization": random_metadata or {},
        "artifacts": {
            "model": f"{args.i}.pth" if getattr(args, "save_attacked_model", False) else None,
            "poison_train": poison_train_path.name if poison_train_path else None,
            "poison_test": poison_test_path.name if poison_test_path else None,
        },
        "args": {
            key: _json_safe(value)
            for key, value in vars(args).items()
            if key not in {"model", "logger", "train_set", "collate_fn"}
        },
    }
    metadata_path = save_folder_path / f"{args.i}_trigger.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    return metadata_path


def _get_save_folder_path(args):
    save_folder_path = BASE_DIR / "data"
    folder_name = f"""{args.data_type}-{args.dataset}-{"benign" if args.train_benign else args.attack_name}-{args.model_name}"""
    return save_folder_path / folder_name


def _has_trained_model(args, model_index):
    """Check whether a specific model index has already been trained and saved."""
    save_folder_path = _get_save_folder_path(args)
    if not save_folder_path.exists():
        return False

    # Priority 1: checkpoint file exists (0.pth, 1.pth, ...)
    if (save_folder_path / f"{model_index}.pth").exists():
        return True

    # Priority 2: result log already has this index
    result_log_path = save_folder_path / "result.log"
    if result_log_path.exists():
        try:
            with open(result_log_path, "r", encoding="utf-8") as f:
                content = f.read()
            if f"第{model_index}个模型" in content:
                return True
        except OSError:
            return False

    return False


def atk_train(args):
    """this is the entry of the attack process

    Args:
        args: Parameters required for the attack
    """

    # set log path
    train_log_path = get_log_path_by_args(
        data_type=args.data_type,
        attack_name=args.attack_name,
        dataset=args.dataset,
        model_name=args.model_name,
        pratio=args.pratio,
        noise=args.add_noise,
        mislabel=args.mislabel,
    )
    # config log
    logger_name = "attack"
    logger = configure_logger(
        name=logger_name, log_file=train_log_path / "training.log", log_level="debug"
    )
    args.logger = logging.getLogger(logger_name)
    args.save_folder_name = train_log_path
    # load data
    DSW, collate_fn = get_data_spec_class_by_args(args, "all")
    poison_ds_path = get_poison_ds_path_by_args(args)
    clean_train_set = load_dataset(args, train=True)
    args.collate_fn = collate_fn
    args.train_set = DSW(clean_train_set)

    # load train data
    logger.info("loading train data")
    Attack = get_attack_by_args(args)
    clean_test_set = None
    poison_train_path = None
    poison_test_path = None
    if args.train_benign:
        train_set_wrapper = DSW(clean_train_set)
        train_log_path = train_log_path / "benign"
        if not train_log_path.exists():
            train_log_path.mkdir()
    else:
        if args.data_type == "image":
            clean_test_set = load_dataset(args, train=False)
            poison_ds_path, poison_train_path, poison_test_path = make_image_batch_poison_artifacts(
                args,
                Attack,
                clean_train_set=clean_train_set,
                clean_test_set=clean_test_set,
            )
        elif not poison_ds_path.exists():
            train_set_wrapper = Attack(clean_train_set, args, mode="train")
            train_set_wrapper.make_and_save_dataset()
            clean_test_set = load_dataset(args, train=False)
            test_set_wrapper = Attack(clean_test_set, args, mode="test", pop=False)
            test_set_wrapper.make_and_save_dataset()
        train_set_wrapper = BadSet(
            benign_set=DSW(clean_train_set),
            poison_set_path=poison_ds_path,
            type=args.data_type,
            dataset=args.dataset,
            num_classes=len(args.classes),
            mislabel=args.mislabel,
            attack=args.attack_name,
            target_label=args.attack_target,
            poison_rate=args.pratio,
            seed=args.random_seed,
            mode="train",
        )
    logger.info("loaded train data")

    # load model
    logger.info("loading model")
    if args.load_poisoned_model == False:
        orig_model = load_model(args)
        # change model
    else:
        orig_model = load_poisoned_model(args)
    args.model = orig_model

    logger.info("model loaded")
    # get data loader using max batch size
    train_loader = get_dataloader(
        dataset=train_set_wrapper,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=_use_fast_host_to_device_path(args),
        persistent_workers=args.num_workers > 0,
    )
    # save args
    final_args_path = train_log_path / "train_args.yaml"
    with open(final_args_path, "w", encoding="utf-8") as f:
        final_args = dict()
        final_args.update(
            {k: str(v) for k, v in args.__dict__.items() if v is not None}
        )
        yaml.safe_dump(final_args, f, default_flow_style=False)
        logger.info(f"train args saved: {final_args_path.as_posix()}")

    logger.info("start training")
    Train = SupervisedLearningTrain(train_loader, args)
    Train.train_model()
    logger.info("training finished")

    # get test data
    logger.info("loading test data")
    test_loader_lst = []
    if clean_test_set is None:
        clean_test_set = load_dataset(args, train=False)
    clean_test_loader = get_dataloader(
        dataset=DSW(clean_test_set),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=_use_fast_host_to_device_path(args),
        persistent_workers=args.num_workers > 0,
    )
    test_loader_lst.append(clean_test_loader)
    if not args.train_benign:
        data_path = poison_ds_path / "{type}_{attack}_poison_{mode}_set.pt".format(
            type=args.data_type, attack=args.attack_name, mode="test"
        )
        poison_test_set = (
            BadSet(
                benign_set=None,
                poison_set_path=poison_ds_path,
                type=args.data_type,
                dataset=args.dataset,
                num_classes=len(args.classes),
                mislabel=args.mislabel,
                attack=args.attack_name,
                target_label=args.attack_target,
                poison_rate=1,
                mode="test",
                pop=args.data_type != "image",
            )
            if args.data_type != "video"
            else torch.load(data_path)
        )
        poison_test_loader = get_dataloader(
            dataset=poison_test_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            shuffle=False,
            pin_memory=_use_fast_host_to_device_path(args),
            persistent_workers=args.num_workers > 0,
        )
        test_loader_lst.append(poison_test_loader)
    logger.info("test data loaded")

    # test
    logger.info("start testing")
    Eval = SupervisedLearningEval(
        clean_testloader=clean_test_loader,
        poison_testloader=(
            clean_test_loader if args.train_benign else poison_test_loader
        ),
        args=args,
    )
    results = Eval.eval_model()
    acc, asr, ra = results
    logger.info(f"acc, asr and ra: {results}")
    logger.info("test finished")
    # save results
    results_path = train_log_path / "attack_result.json"

    attack_result = {"acc,asr,ra": (acc, asr, ra)}
    save_folder_path = _get_save_folder_path(args)
    if not save_folder_path.exists():
        save_folder_path.mkdir()
    if args.save_attacked_model:
        torch.save(args.model.state_dict(), save_folder_path / f"{args.i}.pth")
    with open(save_folder_path / "result.log", "a") as f:
        new_line = f"第{args.i}个模型的训练结果为：acc:{acc} && asr:{asr} && ra:{ra}\n"
        f.write(new_line)
    logger.info("attack_result.json save in: {path}".format(path=results_path))
    if args.data_type == "image" and not args.train_benign:
        save_batch_attack_metadata(
            args,
            save_folder_path,
            getattr(args, "batch_random_metadata", {}),
            poison_train_path=poison_train_path,
            poison_test_path=poison_test_path,
        )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    init_folders()
    parser = argparse.ArgumentParser()
    init_args(parser)
    parser.add_argument(
        "--batch_number",
        type=int,
        default=100,
        help="the number you want to batch generate",
    )
    parser.add_argument(
        "--i",
        type=int,
        help="the index of the batch training models",
    )
    args = parser.parse_args()
    conf_path = get_cfg_path_by_args(args, "attacks")
    add_yaml_to_args(args, conf_path)
    for i in range(args.batch_number):
        args.i = i
        # Resume support: skip already trained indices and continue with remaining ones.
        if _has_trained_model(args, i):
            print(f"[resume] Skip i={i}, found existing checkpoint/result.")
            continue
        if args.train_benign:
            benign_random_params(args)
        elif args.data_type == "image":
            args.batch_random_metadata = randomize_image_attack_params(args)
        atk_train(args)
