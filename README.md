# BackdoorMBTI

This is an official implentation of the paper [BackdoorMBTI: A Backdoor Learning Multimodal Benchmark Tool Kit for Backdoor Defense Evaluation](https://arxiv.org/pdf/2411.11006). This paper has been accepted by KDD 2025 ADS track. MBTI is an open source project expanding the unimodal backdoor learning to a multimodal context, designed to easily accommodate **new and realistic scenarios and tasks**.

The framework:
![framework](./backdoormbti/resources/arch.png)

## Usage

see our documents: https://backdoormbti.readthedocs.io/

## Results
Part of the experimental results can be found in: [results.md](./backdoormbti/resources/results.md)
The experiments (the results will be updated once the experiment finished):
   - The performance exploration under different **clean model accuracy**
   - The performance exploration under different **poison ratio**
   - The performance exploration under different **noise level**

## Citation
If our work is useful for your research, please cite our related backdoor papers as follows:
```
@inproceedings{yu2025backdoormbti,
  title={BackdoorMBTI: A Backdoor Learning Multimodal Benchmark Tool Kit for Backdoor Defense Evaluation},
  author={Yu, Haiyang and Xie, Tian and Gui, Jiaping and Wang, Pengyang and Cheng, Pengzhou and Yi, Ping and Wu, Yue},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1},
  pages={2791--2802},
  year={2025},
  publisher={ACM},
  doi={10.1145/3690624.3709385}
}

@article{yu2025sequential,
  title={Sequential Feature-Based Backdoor Detection in Deep Neural Networks Via Few-Shot Learning},
  author={Yu, Haiyang and Hong, Fan and Xie, Tian and Yi, Ping and Zou, Futai and Wu, Yue},
  journal={IEEE Internet of Things Journal},
  year={2025}
}

@inproceedings{yu2026graph,
  title={Backdoor Defense via Graph-Structured Reinforcement Learning for Targeted Neuron Pruning},
  author={Yu, Haiyang and Li, Nan and Jiang, Haoyu and Yi, Ping and Wu, Yue},
  booktitle={IEEE International Conference on Multimedia and Expo (ICME)},
  year={2026}
}

@inproceedings{xie2026bmprune,
  title={BMPrune: Bidirectional Magnitude-Based Backdoor Pruning with Clean Preservation and Malicious Penalization},
  author={Xie, Tian and Li, Nan and Yu, Haiyang and Jiang, Haoyu and Yi, Ping and Wu, Yue},
  booktitle={IEEE International Conference on Communications: Communication \& Information System Security (ICC)},
  year={2026}
}

@inproceedings{tun2025ocage,
  title={OCAGE: An Input-Level One-Class Backdoor Detection Method Using Feature Map Extraction for DNN},
  author={Tun, Han Lin and Yu, Haiyang and Xie, Tian and Yi, Ping},
  booktitle={2025 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2025},
  publisher={IEEE},
  doi={10.1109/IJCNN64981.2025.11227936}
}

@inproceedings{jiang2024ocgec,
  title={OCGEC: One-Class Graph Embedding Classification for DNN Backdoor Detection},
  author={Jiang, Haoyu and Yu, Haiyang and Li, Nan and Yi, Ping},
  booktitle={2024 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2024},
  publisher={IEEE},
  doi={10.1109/IJCNN60899.2024.10650468}
}
```


## Changelog

See [CHANGELOG.md](./CHANGELOG.md) for release notes and updates.

## Future Work

BackdoorMBTI will be continuously updated to track the lastest advances of backdoor learning.
The implementations of more backdoor methods, as well as their evaluations are on the way. 
**You are welcome to contribute your backdoor methods to BackdoorMBTI.**

We have a clear roadmap for the next phases of BackdoorMBTI development, including:

1. refactoring the training pipeline and defense design, enhancing the code quality.
2. adding more test cases for high usability.
3. adding function comments for ReadTheDocs.
4. fix bug currently found (low accuracy of video training).
5. adding tasks and modalities we promised (VQA and audiovisual).

