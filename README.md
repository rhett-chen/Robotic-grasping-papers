# Robotic Grasping Papers and Codes

This repo is a paper list of Robotic-Grasping and some related works that I have read.  For a more comprehensive paper list of vision-based Robotic-Grasping, you can refer to [Vision-based Robotic Grasping: Papers and Codes](https://github.com/GeorgeDu/vision-based-robotic-grasping) of [Guoguang DU](https://github.com/GeorgeDu).

## Robotic Grasping Papers and Codes

1. [Survey Papers](#survey-papers)

2. [Object Pose Estimation](#object-pose-estimation)

3. [Grasp Detection](#grasp-detection)

4. [Task-oriented Methods](#task-oriented-grasp)

5. [Research groups](#research-groups)

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

**[ICCV2021]** Graspness Discovery in Clutters for Fast and Accurate Grasp Detection,        [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Graspness_Discovery_in_Clutters_for_Fast_and_Accurate_Grasp_Detection_ICCV_2021_paper.pdf)],  [[Code(non-official)](https://github.com/rhett-chen/graspness_implementation)],  [[Dataset: GraspNet1-billion](https://graspnet.net/datasets.html)], [[Supp](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Wang_Graspness_Discovery_in_ICCV_2021_supplemental.zip)].

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

**[ICRA2021]** RGB Matters: Learning 7-DoF Grasp Poses on Monocular RGBD Images,       [[Paper](https://arxiv.org/pdf/2103.02184)].

*Keywords: 6D genral grasp; clutter scene; RGB and single-view point cloud input; real-world dataset GraspNet1-billion.*

```latex
@inproceedings{gou2021RGB,
  title={RGB Matters: Learning 7-DoF Grasp Poses on Monocular RGBD Images},
  author={Minghao Gou, Hao-Shu Fang, Zhanda Zhu, Sheng Xu, Chenxi Wang, Cewu Lu},
  booktitle={Proceedings of the International Conference on Robotics and Automation (ICRA)},
  year={2021}
}
```

**[ICRA2021]** GPR: Grasp Pose Refinement Network for Cluttered Scenes,  [[Paper](https://arxiv.org/pdf/2105.08502)].

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

**[ICRA2021]** Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes,     [[Project](https://research.nvidia.com/publication/2021-03_Contact-GraspNet%3A--Efficient)], [[Paper](https://arxiv.org/pdf/2103.14127)], [[Code](https://github.com/NVlabs/contact_graspnet)].

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

**[ICRA2021]** Acronym: A large-scale grasp dataset based on simulation, [[Project](https://sites.google.com/nvidia.com/graspdataset)],  [[Paper](https://arxiv.org/pdf/2011.09584)], [[Code](https://github.com/NVlabs/acronym)].

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
 url = {https://proceedings.neurips.cc/paper/2020/file/994d1cad9132e48c993d58b492f71fc1-Paper.pdf},
 volume = {33},
 year = {2020}
}
```

**[CoRL2020]** GDN: A Coarse-To-Fine (C2F) Representation for End-To-End 6-DoF Grasp Detection, [[Paper](https://proceedings.mlr.press/v155/jeng21a/jeng21a.pdf)]

*Keywords: 6D grasp; one-stage method; single object; Pointnet++ backbone; self-made dataset based on YCB.*

```latex
@InProceedings{pmlr-v155-jeng21a,
  title =      {GDN: A Coarse-To-Fine (C2F) Representation for End-To-End 6-DoF Grasp Detection},
  author =       {Jeng, Kuang-Yu and Liu, Yueh-Cheng and Liu, Zhe Yu and Wang, Jen-Wei and Chang, Ya-Liang and Su, Hung-Ting and Hsu, Winston},
  booktitle =      {Proceedings of the 2020 Conference on Robot Learning},
  pages =      {220--231},
  year =      {2021},
  editor =      {Kober, Jens and Ramos, Fabio and Tomlin, Claire},
  volume =      {155},
  series =      {Proceedings of Machine Learning Research},
  month =      {16--18 Nov},
  publisher =    {PMLR},
  pdf =      {https://proceedings.mlr.press/v155/jeng21a/jeng21a.pdf},
  url =      {https://proceedings.mlr.press/v155/jeng21a.html},
  abstract =      {We proposed an end-to-end grasp detection network,  Grasp Detection Network (GDN), cooperated with a novel coarse-to-fine (C2F) grasp representation design to detect diverse and accurate 6-DoF grasps based on point clouds.   Compared to previous two-stage approaches which sample and evaluate multiple grasp candidates, our architecture is at least 20 times faster.  It is also 8% and 40% more accurate in terms of the success rate in single object scenes and the complete rate in clutter scenes, respectively. Our method shows superior results among settings with different number of views and input points.  Moreover, we propose a new AP-based metric which considers both rotation and transition errors, making it a more comprehensive evaluation tool for grasp detection models.}
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

## 5. Research groups  <span id="research-groups"> </span>
