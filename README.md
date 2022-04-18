# Robotic Grasping Papers and Codes

This repo is a paper list of Robotic-Grasping and some related works that I have read.  For a more comprehensive paper list of vision-based Robotic-Grasping, you can refer to [Vision-based Robotic Grasping: Papers and Codes](https://github.com/GeorgeDu/vision-based-robotic-grasping) of [Guoguang DU](https://github.com/GeorgeDu).

Abbreviation:  **ICRA** is IEEE International Conference on Robotics and Automation;  **CVPR** is IEEE Conference on Computer Vision and Pattern Recognition;  **ICCV** is IEEE International Conference on Computer Vision; **CoRL** is Conference on Robot Learning; **NIPS** is Conference on Neural Information Processing Systems;  **RA-L** is IEEE Robotics and Automation Letters; **Humanoids** is IEEE-RAS International Conference on Humanoid Robots; **IJRR** is The International Journal of Robotics Research.

## Robotic Grasping Papers and Codes

1. [Survey Papers](#survey-papers)

2. [Object Pose Estimation](#object-pose-estimation)

3. [Grasp Detection](#grasp-detection)

4. [Task-oriented Methods](#task-oriented-grasp)

5. [Semantic grasping](#semantic-grasping)

6. [Research groups](#research-groups)

## 1. Survey Papers <span id="survey-papers"> </span>

**[arXiv2022]** Robotic Grasping from Classical to Modern: A Survey,   [[Project](https://github.com/ZhangHanbo/Robotic-Grasping-from-Classical-to-Modern-A-Survey)],  [[Paper](https://arxiv.org/pdf/2202.03631.pdf)].

*Keywords: overview of analytic methods and data-driven methods for Robotic grasping.*

```latex
@article{zhang2022robotic,
  title={Robotic Grasping from Classical to Modern: A Survey},
  author={Zhang, Hanbo and Tang, Jian and Sun, Shiguang and Lan, Xuguang},
  journal={arXiv preprint arXiv:2202.03631},
  year={2022}
}
```

**[Artifcial Intelligence Review (2021)]** Vision‑based robotic grasping from object localization, object pose estimation to grasp estimation for parallel grippers: a review,   [[Paper](https://link.springer.com/content/pdf/10.1007/s10462-020-09888-5.pdf)].

*Keywords: object localization, object pose estimation and grasp estimation.*

```latex
@article{du2021vision,
  title={Vision-based robotic grasping from object localization, object pose estimation to grasp estimation for parallel grippers: a review},
  author={Du, Guoguang and Wang, Kai and Lian, Shiguo and Zhao, Kaiyong},
  journal={Artificial Intelligence Review},
  volume={54},
  number={3},
  pages={1677--1734},
  year={2021},
  publisher={Springer}
}
```

## 2. Object Pose Estimation <span id="object-pose-estimation"> </span>

## 3. Grasp Detection<span id="grasp-detection"> </span>

**[ICCV2021]** Graspness Discovery in Clutters for Fast and Accurate Grasp Detection,        [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Graspness_Discovery_in_Clutters_for_Fast_and_Accurate_Grasp_Detection_ICCV_2021_paper.pdf)],  [[Code(non-official)](https://github.com/rhett-chen/graspness_implementation)], [[Supp](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Wang_Graspness_Discovery_in_ICCV_2021_supplemental.zip)].

*Keywords: 6D general grasp; clutter scene; real-world dataset GraspNet1-billion;  single-view scene point cloud input; MinkowskiEngine sparse convolution, ResUNet14.*

```latex
@InProceedings{Wang_2021_ICCV,
    author    = {Wang, Chenxi and Fang, Hao-Shu and Gou, Minghao and Fang, Hongjie and Gao, Jin and Lu, Cewu},
    title     = {Graspness Discovery in Clutters for Fast and Accurate Grasp Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {15964-15973}
}
```

**[ICRA2021]** RGB Matters: Learning 7-DoF Grasp Poses on Monocular RGBD Images,       [[Paper](https://arxiv.org/pdf/2103.02184.pdf)].

*Keywords: 6D genral grasp; clutter scene; RGB and single-view point cloud input; real-world dataset GraspNet1-billion.*

```latex
@inproceedings{gou2021RGB,
  title={RGB Matters: Learning 7-DoF Grasp Poses on Monocular RGBD Images},
  author={Minghao Gou, Hao-Shu Fang, Zhanda Zhu, Sheng Xu, Chenxi Wang, Cewu Lu},
  booktitle={Proceedings of the International Conference on Robotics and Automation (ICRA)},
  year={2021}
}
```

**[ICRA2021]** GPR: Grasp Pose Refinement Network for Cluttered Scenes,  [[Paper](https://arxiv.org/pdf/2105.08502.pdf)].

*Keywords: 6D grasp; two-stage method; self-made dataset in simulation; clutter scene in simulation; Pointnet++ backbone.*

```latex
@inproceedings{wei2021gpr,
  title={Gpr: Grasp pose refinement network for cluttered scenes},
  author={Wei, Wei and Luo, Yongkang and Li, Fuyu and Xu, Guangyun and Zhong, Jun and Li, Wanyi and Wang, Peng},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={4295--4302},
  year={2021},
  organization={IEEE}
}
```

**[ICRA2021]** Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes,     [[Project](https://research.nvidia.com/publication/2021-03_Contact-GraspNet%3A--Efficient)], [[Paper](https://arxiv.org/pdf/2103.14127.pdf)], [[Code](https://github.com/NVlabs/contact_graspnet)].

*Keywords: 6D grasp; object grasp dataset is ACRONYM; clutter scene in simulation; single-view scene point cloud(20000 points) input; backbone based on Pointnet++.*

```latex
@inproceedings{sundermeyer2021contact,
  title={Contact-graspnet: Efficient 6-dof grasp generation in cluttered scenes},
  author={Sundermeyer, Martin and Mousavian, Arsalan and Triebel, Rudolph and Fox, Dieter},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={13438--13444},
  year={2021},
  organization={IEEE}
}
```

**[ICRA2021]** Acronym: A large-scale grasp dataset based on simulation, [[Project](https://sites.google.com/nvidia.com/graspdataset)],   [[Paper](https://arxiv.org/pdf/2011.09584.pdf)], [[Code](https://github.com/NVlabs/acronym)].

*Keywords: 6D grasp; release a grasp dataset in simulation.*

```latex
@inproceedings{eppner2021acronym,
  title={ACRONYM: A large-scale grasp dataset based on simulation},
  author={Eppner, Clemens and Mousavian, Arsalan and Fox, Dieter},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={6222--6227},
  year={2021},
  organization={IEEE}
}
```

**[CVPR2020]**  GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping,  [[Project](https://graspnet.net/index.html)],  [[Paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.pdf)],  [[Code](https://github.com/graspnet)], [[Supp](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Fang_GraspNet-1Billion_A_Large-Scale_CVPR_2020_supplemental.pdf)].

*Keywords: 6D general grasp; release a large-scale real-world dataset; clutter scene; single-view scene point cloud input; pointnet++ backbone.*

```latex
@inproceedings{fang2020graspnet,
  title={GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping},
  author={Fang, Hao-Shu and Wang, Chenxi and Gou, Minghao and Lu, Cewu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11444--11453},
  year={2020}
}
```

**[NIPS2020]** Grasp Proposal Networks: An End-to-End Solution for Visual Learning of Robotic Grasps, [[Paper](https://proceedings.neurips.cc/paper/2020/file/994d1cad9132e48c993d58b492f71fc1-Paper.pdf)], [[Code](https://github.com/CZ-Wu/GPNet)], [[Supp](https://proceedings.neurips.cc/paper/2020/file/994d1cad9132e48c993d58b492f71fc1-Supplemental.zip)].

*Keywords: 6D grasp; single object; single-view point cloud input; Pointnet++ backbone; self-made synthetic dataset based on ShapeNetSem in simulation.*

```latex
@inproceedings{NEURIPS2020_994d1cad,
 author = {Wu, Chaozheng and Chen, Jian and Cao, Qiaoyu and Zhang, Jianchi and Tai, Yunxin and Sun, Lin and Jia, Kui},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {13174--13184},
 publisher = {Curran Associates, Inc.},
 title = {Grasp Proposal Networks: An End-to-End Solution for Visual Learning of Robotic Grasps},
 volume = {33},
 year = {2020}
}
```

**[CoRL2020]** GDN: A Coarse-To-Fine (C2F) Representation for End-To-End 6-DoF Grasp Detection, [[Paper](https://proceedings.mlr.press/v155/jeng21a/jeng21a.pdf)].

*Keywords: 6D grasp; one-stage method; single object; Pointnet++ backbone; self-made dataset based on YCB.*

```latex
@InProceedings{pmlr-v155-jeng21a,
  title =      {GDN: A Coarse-To-Fine (C2F) Representation for End-To-End 6-DoF Grasp Detection},
  author =       {Jeng, Kuang-Yu and Liu, Yueh-Cheng and Liu, Zhe Yu and Wang, Jen-Wei and Chang, Ya-Liang and Su, Hung-Ting and Hsu, Winston},
  booktitle =      {Proceedings of the 2020 Conference on Robot Learning},
  pages =      {220--231},
  year =      {2021},
  volume =      {155},
  series =      {Proceedings of Machine Learning Research},
  month =      {16--18 Nov},
  publisher =    {PMLR},
}
```

## 4.Task-oriented Grasp  <span id="task-oriented-grasp"> </span>

**[ICRA2022]** CaTGrasp: Learning Category-Level Task-Relevant Grasping in Clutter from Simulation,  [[Project](https://sites.google.com/view/catgrasp)],  [[Paper](https://arxiv.org/pdf/2109.09163v1.pdf)],  [[Code](https://github.com/wenbowen123/catgrasp)].  

*Keywords: 6D category-level task-oriented grasp; clutter scene in simulation;  self-supervised in simulation.*

- [ ] Bibtex to be updated

```latex
@article{wen2021catgrasp,
  title={CaTGrasp: Learning Category-Level Task-Relevant Grasping in Clutter from Simulation},
  author={Wen, Bowen and Lian, Wenzhao and Bekris, Kostas and Schaal, Stefan},
  journal={arXiv preprint arXiv:2109.09163},
  year={2021}
}
```

**[RA-L2022]** GATER: Learning Grasp-Action-Target Embeddings and Relations for Task-Specific Grasping,  [[Paper](https://arxiv.org/pdf/2111.13815.pdf)].

*Keywords: 2D planar grasp; task-oriented grasp; self-made task-oriented grasp dataset; grasp-action-target relationship.*

```latex
@ARTICLE{9629256,
  author={Sun, Ming and Gao, Yue},
  journal={IEEE Robotics and Automation Letters}, 
  title={GATER: Learning Grasp-Action-Target Embeddings and Relations for Task-Specific Grasping}, 
  year={2022},
  volume={7},
  number={1},
  pages={618-625},
  doi={10.1109/LRA.2021.3131378}}{IEEE}
}
```

**[CoRL2020]** Same Object, Different Grasps: Data and SemanticKnowledge for Task-Oriented Grasping, [[Project](https://sites.google.com/view/taskgrasp)], [[Paper](https://proceedings.mlr.press/v155/murali21a/murali21a.pdf)], [[Code](https://github.com/adithyamurali/TaskGrasp)], [[Dataset](https://drive.google.com/file/d/1aZ0k43fBIZZQPPPraV-z6itpqCHuDiUU/view)].

*Keywords: 6D task-oriented grasp; single object; real-world data; object point cloud and goal task input; Pointnet++ backbone for point cloud,  Graph Convolutional Network for object and task semantic knowledge.*

```latex
@inproceedings{murali2020taskgrasp,
  title={Same Object, Different Grasps: Data and Semantic Knowledge for Task-Oriented Grasping},
  author={Murali, Adithyavairavan and Liu, Weiyu and Marino, Kenneth and Chernova, Sonia and Gupta, Abhinav},
  booktitle={Conference on Robot Learning},
  year={2020}
}
```

**[IJRR2020]** Learning Task-Oriented Grasping for Tool Manipulation from Simulated Self-Supervision, [[Project](https://sites.google.com/view/task-oriented-grasp/)], [[Paper](https://arxiv.org/pdf/1806.09266.pdf)].

Keywords: task-oriented grasping; manipulation policy; self-supervised in simulation; single object; planar grasp; depth image input; 2 tasks.

```latex
@article{fang2020learning,
  title={Learning task-oriented grasping for tool manipulation from simulated self-supervision},
  author={Fang, Kuan and Zhu, Yuke and Garg, Animesh and Kurenkov, Andrey and Mehta, Viraj and Fei-Fei, Li and Savarese, Silvio},
  journal={The International Journal of Robotics Research},
  volume={39},
  number={2-3},
  pages={202--216},
  year={2020},
  publisher={SAGE Publications Sage UK: London, England}
}
```

**[ICRA2018]** AffordanceNet: An End-to-End Deep Learning Approach for Object Affordance Detection, [[Paper](https://www.csc.liv.ac.uk/~anguyen/assets/pdfs/2018_ICRA_AffordanceNet.pdf)], [[Code](https://github.com/nqanh/affordance-net)].

*Keywords: simultaneous object detection and affordance detection; RGB input.*

```latex
@inproceedings{do2018affordancenet,
  title={Affordancenet: An end-to-end deep learning approach for object affordance detection},
  author={Do, Thanh-Toan and Nguyen, Anh and Reid, Ian},
  booktitle={2018 IEEE international conference on robotics and automation (ICRA)},
  pages={5882--5889},
  year={2018},
  organization={IEEE}
}
```

**[Humanoids2017]** Affordance Detection for Task-Specific Grasping Using Deep Learning,  [[Paper](https://www.cs.columbia.edu/~allen/S19/Student_Papers/kragic_affordance_grasp_planning.pdf)].

*keywords: single object point cloud and task name input; output affordance detection, no grasp; 5 tasks, 10 object classes; generalize to novel object class.*

```latex
@inproceedings{kokic2017affordance,
  title={Affordance detection for task-specific grasping using deep learning},
  author={Kokic, Mia and Stork, Johannes A and Haustein, Joshua A and Kragic, Danica},
  booktitle={2017 IEEE-RAS 17th International Conference on Humanoid Robotics (Humanoids)},
  pages={91--98},
  year={2017},
  organization={IEEE}
}
```

## 5. Semantic grasping <span id="semantic-grasping"> </span>

**[arXiv2021]** StructFormer: Learning Spatial Structurefor Language-Guided Semantic Rearrangement of Novel Objects, [[Paper](https://arxiv.org/pdf/2110.10189)].

*Keywords: language-guided semantic rearrangement; transformed-based method; scene point cloud and structured language command input; output plan sequence, no 6D grasp.*

```latex
@article{liu2021structformer,
  title={Structformer: Learning spatial structure for language-guided semantic rearrangement of novel objects},
  author={Liu, Weiyu and Paxton, Chris and Hermans, Tucker and Fox, Dieter},
  journal={arXiv preprint arXiv:2110.10189},
  year={2021}
}
```

## 6. Research groups  <span id="research-groups"> </span>

- [SJTU Machine Vision and Intelligence Group](https://mvig.sjtu.edu.cn/),  Professor Cewu Lu [[Google Scholar](https://scholar.google.de/citations?user=QZVQEWAAAAAJ&hl=zh-CN&oi=ao)].
