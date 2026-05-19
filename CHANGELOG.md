# Changelog

## 2026-05-19

1. Update `README.md` citation entries with related backdoor papers.
2. Move changelog entries from `README.md` to `CHANGELOG.md`.

## 2026-03-18

1. Add resume function for batch_train script.

## 2026-03-12

1. Update the poison generation script in `resources/bite`.
2. Add `README.md` file for bite data preparation.
3. Fix bug in preprocessing for model LSTM.
4. Add multi-GPU serial training bash in `backdoormbti/batch_train`.
5. Fix bug in `atk_batch_train.py`.

## 2025-08-17

1. Update the batch_train scripts.
2. Fix bug in `README.md`, update the citation.

## 2025-06-30

1. Update the docs.
2. Fix bug of args name in atk_train.
3. Update the link in `README.md`.
4. Format using black.

## 2024-12-11

We are actively developing BackdoorMBTI with exciting new features recently added:

1. **Tasks/Modalities**: we have added 2 new type of tasks (video and contrasive learning), we are trying to support VQA task and fix the bug in R3D learning.

2. **Models**: we are now support 22 models in total (including ViT, RoBERTa, GPT2, X-Vector, HuBERT, R3D).

3. **New Attacks:** we are adding 19 attacks (9 image, 1 text, 1 audio, 4 video, 1 audiovisual, 1 contrasive learning, 2 visual question answering), see section [attacks](https://backdoormbti.readthedocs.io/en/latest/tutorials/attacks.html) for detail.

4. **New Defenses:** we have added 2 backdoor defense method (MNTD and FreeEagle), see section [defenses](https://backdoormbti.readthedocs.io/en/latest/tutorials/defenses.html) for detail.

5. **Documentation**: we have set up our documentation pages at [https://backdoormbti.readthedocs.io/](https://backdoormbti.readthedocs.io/).

6. **PyPI Package**: we have packaged BackdoorMBTI as a PyPI package for easier installation and integration.

7. **Results**: we are running serveral experiments, new results will be updated at [results.md](./backdoormbti/resources/results.md)

8. **CI Pipeline**: we have added CI pipeline for premerge test.

9. **Test Cases**: we have added first test case for BadNets.
