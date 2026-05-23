import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

BACKDOORMBTI_DIR = Path(__file__).resolve().parents[1] / "backdoormbti"
if str(BACKDOORMBTI_DIR) not in sys.path:
    sys.path.insert(0, str(BACKDOORMBTI_DIR))

from batch_train import atk_batch_train


def _args(**overrides):
    values = {
        "data_type": "image",
        "dataset": "cifar10",
        "attack_name": "blend",
        "model_name": "resnet18",
        "train_benign": False,
        "i": 0,
        "random_seed": 7,
        "num_classes": 10,
        "attack_target": 0,
        "pratio": 0.1,
        "attack_trigger_img_path": "resources/blend/hello_kitty.jpeg",
        "patch_mask_path": "resources/badnet/trigger_image.png",
        "trigger_path": "resources/trojan/trigger.png",
        "attack_train_blended_alpha": 0.2,
        "save_attacked_model": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_image_attack_randomization_is_seeded_and_keeps_trigger_paths():
    first = _args(i=1)
    second = _args(i=1)

    first_meta = atk_batch_train.randomize_image_attack_params(first)
    second_meta = atk_batch_train.randomize_image_attack_params(second)

    assert first_meta == second_meta
    assert 0 <= first.attack_target < first.num_classes
    assert 0.05 <= first.pratio <= 0.5
    assert 0.05 <= first_meta["mmt_reference"]["inject_p"] <= 0.5
    assert first.attack_trigger_img_path == "resources/blend/hello_kitty.jpeg"
    assert first.patch_mask_path == "resources/badnet/trigger_image.png"
    assert first.trigger_path == "resources/trojan/trigger.png"
    assert first.attack_train_blended_alpha == first_meta["randomized_params"]["attack_train_blended_alpha"]


def test_randomization_only_updates_params_read_by_current_attack():
    args = _args(attack_name="sig", poison_type="sin")

    meta = atk_batch_train.randomize_image_attack_params(args)

    assert args.poison_type in {"sin", "ramp", "triangle"}
    assert "poison_type" in meta["randomized_params"]
    assert "attack_train_blended_alpha" not in meta["randomized_params"]


def test_batch_attack_metadata_is_saved_next_to_model_artifacts(tmp_path):
    args = _args(i=3)
    metadata = atk_batch_train.randomize_image_attack_params(args)
    train_pt = tmp_path / "3_poison_train.pt"
    test_pt = tmp_path / "3_poison_test.pt"
    train_pt.write_bytes(b"train")
    test_pt.write_bytes(b"test")

    path = atk_batch_train.save_batch_attack_metadata(
        args,
        tmp_path,
        metadata,
        poison_train_path=train_pt,
        poison_test_path=test_pt,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path == tmp_path / "3_trigger.json"
    assert payload["model_index"] == 3
    assert payload["attack_name"] == "blend"
    assert payload["attack_target"] == args.attack_target
    assert payload["pratio"] == args.pratio
    assert payload["trigger_paths"]["attack_trigger_img_path"] == "resources/blend/hello_kitty.jpeg"
    assert payload["artifacts"]["poison_train"] == "3_poison_train.pt"
    assert payload["artifacts"]["poison_test"] == "3_poison_test.pt"


class _FakeAttack:
    attack_type = "image"
    attack_name = "badnet"

    def __init__(self, dataset, args, mode="train", pop=True):
        self.dataset = dataset
        self.args = args
        self.mode = mode

    def make_and_save_dataset(self, save_dir=None):
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save([(torch.zeros(1), 0, 1, 1)], save_dir / f"image_badnet_poison_{self.mode}_set.pt")


def test_image_poison_artifacts_are_snapshotted_per_model(tmp_path, monkeypatch):
    args = _args(attack_name="badnet", i=2)
    monkeypatch.setattr(atk_batch_train, "_get_save_folder_path", lambda _: tmp_path)

    poison_dir, train_path, test_path = atk_batch_train.make_image_batch_poison_artifacts(
        args,
        _FakeAttack,
        clean_train_set=[(torch.zeros(1), 1)],
        clean_test_set=[(torch.zeros(1), 1)],
    )

    assert poison_dir == tmp_path / "2_poison_data"
    assert train_path == tmp_path / "2_poison_train.pt"
    assert test_path == tmp_path / "2_poison_test.pt"
    assert train_path.exists()
    assert test_path.exists()
    assert (poison_dir / "image_badnet_poison_train_set.pt").exists()
    assert (poison_dir / "image_badnet_poison_test_set.pt").exists()


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_BATCH_TRAIN_INTEGRATION") != "1",
    reason="set RUN_BATCH_TRAIN_INTEGRATION=1 to run the 2-epoch/2-model batch training smoke test",
)
def test_batch_image_training_smoke_generates_two_models_with_artifacts():
    import subprocess

    command = [
        sys.executable,
        "backdoormbti/batch_train/atk_batch_train.py",
        "--data_type",
        "image",
        "--dataset",
        "cifar10",
        "--attack_name",
        "badnet",
        "--model_name",
        "resnet18",
        "--epochs",
        "2",
        "--batch_number",
        "2",
        "--batch_size",
        "16",
        "--num_workers",
        "0",
        "--fast_dev",
    ]
    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True, timeout=600)

    artifact_dir = Path("backdoormbti/data/image-cifar10-badnet-resnet18")
    assert (artifact_dir / "0.pth").exists()
    assert (artifact_dir / "1.pth").exists()
    assert (artifact_dir / "0_trigger.json").exists()
    assert (artifact_dir / "1_trigger.json").exists()
    assert (artifact_dir / "0_poison_train.pt").exists()
    assert (artifact_dir / "1_poison_train.pt").exists()
