# Robotic Grasping Papers and Codes

This repo is a paper list of Robotic-Grasping and some related works(6D pose estimation, 3D visual grounding, etc) that I have read.  For a more comprehensive paper list of vision-based Robotic-Grasping, you can refer to [Vision-based Robotic Grasping: Papers and Codes](https://github.com/GeorgeDu/vision-based-robotic-grasping) of [Guoguang DU](https://github.com/GeorgeDu).

Abbreviation:  **ICRA** is IEEE International Conference on Robotics and Automation;  **CVPR** is IEEE Conference on Computer Vision and Pattern Recognition;  **ICCV** is IEEE International Conference on Computer Vision; **CoRL** is Conference on Robot Learning; **NIPS** is Conference on Neural Information Processing Systems;  **RA-L** is IEEE Robotics and Automation Letters; **Humanoids** is IEEE-RAS International Conference on Humanoid Robots; **IJRR** is The International Journal of Robotics Research; **IROS** is IEEE/RSJ International Conference on Intelligent Robots and Systems; **ACM MM** is  ACM International Conference on Multimedia.

## Robotic Grasping Papers and Codes

1. [Survey Papers](#survey-papers)

2. [Related Vision Tasks](#related-vision-tasks)
   
   2.1 [Visual grounding](#visual-grounding)   
   
   2.2 [Robotic Manipulation](#robotic-manipulation)
   
   2.3 [6D pose estimation](#6d-pose-estimation)

3. [Grasp Detection](#grasp-detection)
   
   3.1 [General grasping](#general-grasping)
   
   3.2 [Semantic grasping](#semantic-grasping)

4. [Research Groups](#research-groups)

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

## 2. Related Vision Tasks <span id="related-vision-tasks"> </span>

### 2.1 Visual grounding <span id="visual-grounding"> </span>

**[CVPR2022]** Multi-View Transformer for 3D Visual Grounding, [[Paper](https://arxiv.org/pdf/2204.02174.pdf)], [[Code](https://github.com/sega-hsj/MVT-3DVG)].

*Keywords: Transformer based; learn view-robust representation, eliminate the dependence on specific views.*

```latex
@InProceedings{Huang_2022_CVPR,
    author    = {Huang, Shijia and Chen, Yilun and Jia, Jiaya and Wang, Liwei},
    title     = {Multi-View Transformer for 3D Visual Grounding},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {15524-15533}
}
```

**[CVPR2022]** 3D-SPS: Single-Stage 3D Visual Grounding via Referred Point Progressive Selection, [[Paper](https://arxiv.org/pdf/2204.06272.pdf)], [[Code](https://github.com/fjhzhixi/3D-SPS)].

*Keywords: input point cloud, RGB, normal vectors and language text; PointNet++ backbone for point cloud; output target object bounding box; single-stage method; cross-modal transformer model is used.*

```latex
@InProceedings{Luo_2022_CVPR,
    author    = {Luo, Junyu and Fu, Jiahui and Kong, Xianghao and Gao, Chen and Ren, Haibing and Shen, Hao and Xia, Huaxia and Liu, Si},
    title     = {3D-SPS: Single-Stage 3D Visual Grounding via Referred Point Progressive Selection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {16454-16463}
}
```

**[CoRL2021]** LanguageRefer: Spatial-Language Model for 3D Visual Grounding, [[Project](https://sites.google.com/view/language-refer)], [[Paper](https://proceedings.mlr.press/v164/roh22a/roh22a.pdf)], [[Code](https://github.com/rohjunha/language-refer)].

*Keywords: 3D scene point clouds; ReferIt3D dataset; transformer-based network; add viewpoint annotation, has explicit viewpoint information in the utterance.*

```latex
@InProceedings{pmlr-v164-roh22a,
  title =      {LanguageRefer: Spatial-Language Model for 3D Visual Grounding},
  author =       {Roh, Junha and Desingh, Karthik and Farhadi, Ali and Fox, Dieter},
  booktitle =      {Proceedings of the 5th Conference on Robot Learning},
  pages =      {1046--1056},
  year =      {2022},
  volume =      {164},
  series =      {Proceedings of Machine Learning Research},
  publisher = {PMLR},
}
```

**[CoRL2021]** Language Grounding with 3D Objects, [[Paper](https://proceedings.mlr.press/v164/thomason22a/thomason22a.pdf)], [[Code](https://github.com/snaredataset/snare)], [[Supp](https://proceedings.mlr.press/v164/thomason22a/thomason22a-supp.zip)].

*Keywords: distinguish between object pair based on object referring expressions; annotated SNARE Dataset, based on ShapNet, 7897 objects, 50000 natural language referring expressions.*

```latex
@inproceedings{thomason2022language,
  title={Language grounding with 3D objects},
  author={Thomason, Jesse and Shridhar, Mohit and Bisk, Yonatan and Paxton, Chris and Zettlemoyer, Luke},
  booktitle={Conference on Robot Learning},
  pages={1691--1701},
  year={2022},
  organization={PMLR}
}
```

**[CVPR2021]** Refer-it-in-RGBD: A Bottom-up Approach for 3D Visual Grounding in RGBD Images, [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Refer-It-in-RGBD_A_Bottom-Up_Approach_for_3D_Visual_Grounding_in_RGBD_CVPR_2021_paper.pdf)], [[Code](https://github.com/UncleMEDM/Refer-it-in-RGBD)], [[Supp](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Liu_Refer-It-in-RGBD_A_Bottom-Up_CVPR_2021_supplemental.pdf)].

*Keywords: new task, single view 3D visual grounding; input single-view RGBD and language text; fuse language and visual feature gradually; contribute a large-scale dataset by annotating SUNRGBD with referring expressions; GLoVe for word embedding; use GRU to encode description.*

```latex
@InProceedings{Liu_2021_CVPR,
    author    = {Liu, Haolin and Lin, Anran and Han, Xiaoguang and Yang, Lei and Yu, Yizhou and Cui, Shuguang},
    title     = {Refer-It-in-RGBD: A Bottom-Up Approach for 3D Visual Grounding in RGBD Images},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {6032-6041}
}
```

**[ICCV2021]** SAT: 2D Semantics Assisted Training for 3D Visual Grounding, [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Yang_SAT_2D_Semantics_Assisted_Training_for_3D_Visual_Grounding_ICCV_2021_paper.pdf)], [[Code](https://github.com/zyang-ur/SAT)], [[Supp](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Yang_SAT_2D_Semantics_ICCV_2021_supplemental.pdf)].

*Keywords: 2D image assisted training, don't need 2D image in inference; auxiliary loss functions that align objects in 2D images with the corresponding ones in 3D point clouds or language queries; transformer-based method.*

```latex
@InProceedings{Yang_2021_ICCV,
    author    = {Yang, Zhengyuan and Zhang, Songyang and Wang, Liwei and Luo, Jiebo},
    title     = {SAT: 2D Semantics Assisted Training for 3D Visual Grounding},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {1856-1866}
}
```

**[ICCV2021]** Free-form Description Guided 3D Visual Graph Network for Object Grounding in Point Cloud, [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Feng_Free-Form_Description_Guided_3D_Visual_Graph_Network_for_Object_Grounding_ICCV_2021_paper.pdf)], [[Code](https://github.com/PNXD/FFL-3DOG)], [[Supp](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Feng_Free-Form_Description_Guided_ICCV_2021_supplemental.pdf)].

*Keywords: free-form description and scene point cloud input; ScanRefer and Nr3D dataset; construct language scene graph and multi-level proposal relation graph; VoteNet for 3D object proposal; GLoVe for word embedding; use GRU to encode description.*

```latex
@InProceedings{Feng_2021_ICCV,
    author    = {Feng, Mingtao and Li, Zhen and Li, Qi and Zhang, Liang and Zhang, XiangDong and Zhu, Guangming and Zhang, Hui and Wang, Yaonan and Mian, Ajmal},
    title     = {Free-Form Description Guided 3D Visual Graph Network for Object Grounding in Point Cloud},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {3722-3731}
}
```

**[ICCV2021]** 3DVG-Transformer: Relation Modeling for Visual Grounding on Point Clouds, [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_3DVG-Transformer_Relation_Modeling_for_Visual_Grounding_on_Point_Clouds_ICCV_2021_paper.pdf)].

*Keywords: transformer based model; grounding by detection; model proposal relation to generate context-aware object proposals; leverage proposal relations to distinguish the true target object from similar proposals.*

```latex
@InProceedings{Zhao_2021_ICCV,
    author    = {Zhao, Lichen and Cai, Daigang and Sheng, Lu and Xu, Dong},
    title     = {3DVG-Transformer: Relation Modeling for Visual Grounding on Point Clouds},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {2928-2937}
}
```

**[ICCV2021]** InstanceRefer: Cooperative Holistic Understanding for Visual Grounding on
Point Clouds through Instance Multi-level Contextual Referring, [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Yuan_InstanceRefer_Cooperative_Holistic_Understanding_for_Visual_Grounding_on_Point_Clouds_ICCV_2021_paper.pdf)], [[Code](https://github.com/CurryYuan/InstanceRefer)].

*Keywords: ScanRefer and ReferIt3D dataset; two-stage method, grounding-by-matching.*

```latex
@InProceedings{yuan2021instancerefer,
  title={Instancerefer: Cooperative holistic understanding for visual grounding on point clouds through instance multi-level contextual referring},
  author={Yuan, Zhihao and Yan, Xu and Liao, Yinghong and Zhang, Ruimao and Li, Zhen and Cui, Shuguang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1791-1800},
  year={2021}
}
```

**[ACM MM2021]** TransRefer3D: Entity-and-Relation Aware Transformer for Fine-Grained 3D Visual Grounding, [[Paper](https://arxiv.org/pdf/2108.02388.pdf)], [[Code](https://github.com/luo-junyu/TransRefer3D)].

*Keywords: transformer-based model; pointnet++ for point cloud; ReferIt3D dataset; entity-aware attention and relation-aware attention for cross-modal feature matching; two auxiliary tasks, utterance classification of the referent and object classification for better feature extraction.*

```latex
@inproceedings{transrefer3d,
    title={TransRefer3D: Entity-and-Relation Aware Transformer for Fine-Grained 3D Visual Grounding},
    author={He, Dailan and Zhao, Yusheng and Luo, Junyu and Hui, Tianrui and Huang, Shaofei and Zhang, Aixi and Liu, Si},
    booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
    year={2021}
}
```

### 2.2 Robotic manipulation<span id="robotic-manipulation"> </span>

**[arXiv2022]** CALVIN: A Benchmark for Language-conditioned Policy Learning for Long-horizon Robot Manipulation Tasks, [[Paper](https://arxiv.org/pdf/2112.03227.pdf)], [[Code](https://github.com/mees/calvin)].

*Keywords: language conditioned long-horizon manipulation; 34 tasks; 4 simulation environments; 7-Dof Panda robot; a static camera and a robot gripper camera; RGB-D image; unstructured demonstrations datasets, ∼2.4M interaction steps.*

```latex
@article{calvin21,
author = {Oier Mees and Lukas Hermann and Erick Rosete-Beas and Wolfram Burgard},
title = {CALVIN: A benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks},
journal={arXiv preprint arXiv:2112.03227},
year = 2021,
}
```

**[CVPR2021]** ManipulaTHOR: A Framework for Visual Object Manipulation, [[Project](https://ai2thor.allenai.org/manipulathor/)], [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Ehsani_ManipulaTHOR_A_Framework_for_Visual_Object_Manipulation_CVPR_2021_paper.pdf)], [[Code](https://github.com/allenai/manipulathor/)], [[Dataset](https://github.com/allenai/manipulathor/tree/main/datasets)].

*Keywords: visual navigvation and object manipulation; simulation environment; dataset includes 30 kitchen scenes, 150+ object categories; sensors include RGB-D, GPS, agent's location and arm configuration.*

```latex
@inproceedings{ehsani2021manipulathor,
  title={Manipulathor: A framework for visual object manipulation},
  author={Ehsani, Kiana and Han, Winson and Herrasti, Alvaro and VanderBilt, Eli and Weihs, Luca and Kolve, Eric and Kembhavi, Aniruddha and Mottaghi, Roozbeh},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={4497--4506},
  year={2021}
}
```

**[arXiv2021]** Audio-Visual Grounding Referring Expression for Robotic Manipulation, [[Paper](https://arxiv.org/pdf/2109.10571.pdf)].

*Keywords: a novel task, audio-visual grounding referring expression for robotic manipulation; establishe a dataset which contains visual data, auditory data and manipulation instructions.*

```latex
@article{wang2021audio,
  title={Audio-Visual Grounding Referring Expression for Robotic Manipulation},
  author={Wang, Yefei and Wang, Kaili and Wang, Yi and Guo, Di and Liu, Huaping and Sun, Fuchun},
  journal={arXiv preprint arXiv:2109.10571},
  year={2021}
}
```

**[arXiv2021]** StructFormer: Learning Spatial Structurefor Language-Guided Semantic Rearrangement of Novel Objects, [[Paper](https://arxiv.org/pdf/2110.10189.pdf)].

*Keywords: language-guided semantic rearrangement; transformer-based method; scene point cloud and structured language command input; output plan sequence, no 6D grasp.*

```latex
@article{liu2021structformer,
  title={Structformer: Learning spatial structure for language-guided semantic rearrangement of novel objects},
  author={Liu, Weiyu and Paxton, Chris and Hermans, Tucker and Fox, Dieter},
  journal={arXiv preprint arXiv:2110.10189},
  year={2021}
}
```

### 2.3 6D pose estimation<span id="6d-pose-estimation"> </span>

**[RA-L2022]** Estimating 6D Object Poses with Temporal Motion Reasoning for Robot Grasping in Cluttered Scenes, [[Paper](https://ieeexplore.ieee.org/abstract/document/9699040/)], [[Code](https://github.com/mufengjun260/H-MPose)].

*Keywords: multi-frame RGB-D sequences; YCB-Video dataset; temporal fusion, integrate the temporal motion information from RGB-D; predict stable pose sequences; handle heavy occlusion.*

```latex
@article{huang2022estimating,
  title={Estimating 6D Object Poses with Temporal Motion Reasoning for Robot Grasping in Cluttered Scenes},
  author={Huang, Rui and Mu, Fengjun and Li, Wenjiang and Liu, Huaping and Cheng, Hong},
  journal={IEEE Robotics and Automation Letters},
  year={2022},
  publisher={IEEE}
}
```

**[CVPR2019]** Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation, [[Project](https://geometry.stanford.edu/projects/NOCS_CVPR2019/)], [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Normalized_Object_Coordinate_Space_for_Category-Level_6D_Object_Pose_and_CVPR_2019_paper.pdf)], [[Code](https://github.com/hughw19/NOCS_CVPR2019)], [[Supp](https://openaccess.thecvf.com/content_CVPR_2019/supplemental/Wang_Normalized_Object_Coordinate_CVPR_2019_supplemental.zip)].

*Keywords: category-level 6D pose estimation; RGB-D image input; CAMERA25 and REAL275 datasets, 6 categories;*

```latex
@InProceedings{Wang_2019_CVPR,
author = {Wang, He and Sridhar, Srinath and Huang, Jingwei and Valentin, Julien and Song, Shuran and Guibas, Leonidas J.},
title = {Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```

**[RSS2018]** PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes, [[Project](https://rse-lab.cs.washington.edu/projects/posecnn/)], [[Paper](https://rse-lab.cs.washington.edu/papers/posecnn_rss18.pdf)], [[Code](https://github.com/yuxng/PoseCNN)].

*Keywords: RGB image input; object segmentation and 6D pose output;  release YCB-video dataset, 21 objects, 92 videos, 133827 frames.*

```latex
@inproceedings{xiang2018posecnn,
    Author = {Xiang, Yu and Schmidt, Tanner and Narayanan, Venkatraman and Fox, Dieter},
    Title = {PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes},
    Journal   = {Robotics: Science and Systems (RSS)},
    Year = {2018}
}
```

## 3. Grasp Detection<span id="grasp-detection"> </span>

### 3.1 General grasping<span id="general-grasping"> </span>

**[RA-L2022]** Real-Time Collision-Free Grasp Pose Detection With Geometry-Aware Refinement Using High-Resolution Volume, [[Project](https://sites.google.com/view/vpn-icra2022)], [[Paper](https://ieeexplore.ieee.org/abstract/document/9681231)].

*Keywords: 6D grasp; cluttered scene; multi-frame depth maps are integrated to get TSDF volume; use a light-weight volume-point network to extract 3D features.*

```latex
@ARTICLE{9681231,
  author={Cai, Junhao and Cen, Jun and Wang, Haokun and Wang, Michael Yu},
  journal={IEEE Robotics and Automation Letters}, 
  title={Real-Time Collision-Free Grasp Pose Detection With Geometry-Aware Refinement Using High-Resolution Volume}, 
  year={2022},
  volume={7},
  number={2},
  pages={1888-1895},
  doi={10.1109/LRA.2022.3142424}}
```

**[arXiv2022]** When Transformer Meets Robotic Grasping: Exploits Context for Efficient Grasp Detection, [[Paper](https://arxiv.org/pdf/2202.11911.pdf)], [[Code](https://github.com/WangShaoSUN/grasp-transformer)].

*Keywords: 2D planar grasp; Cornell and Jacquard grasping datasets; cluttered scene; Transformer based architecture; D/RGB/RGB-D input.*

```latex
@article{wang2022transformer,
  title={When Transformer Meets Robotic Grasping: Exploits Context for Efficient Grasp Detection},
  author={Wang, Shaochen and Zhou, Zhangli and Kan, Zhen},
  journal={arXiv preprint arXiv:2202.11911},
  year={2022}
}
```

**[ICCV2021]** Graspness Discovery in Clutters for Fast and Accurate Grasp Detection,        [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Graspness_Discovery_in_Clutters_for_Fast_and_Accurate_Grasp_Detection_ICCV_2021_paper.pdf)],  [[Code(non-official)](https://github.com/rhett-chen/graspness_implementation)], [[Supp](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Wang_Graspness_Discovery_in_ICCV_2021_supplemental.zip)].

*Keywords: 6D general grasp; cluttered scene; real-world dataset GraspNet1-billion;  single-view scene point cloud input; MinkowskiEngine sparse convolution, ResUNet14.*

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

**[IROS2021]** Simultaneous Semantic and Collision Learning for 6-DoF Grasp Pose Estimation, [[paper](https://arxiv.org/pdf/2108.02425.pdf)].

*Keywords: 6D grasp; cluttered scene; single-view scene point cloud input; real-world dataset GraspNet1-billion; jointly predict grasp poses, semantic segmentation and collision detection.*

```latex
@inproceedings{li2021simultaneous,
  title={Simultaneous Semantic and Collision Learning for 6-DoF Grasp Pose Estimation},
  author={Li, Yiming and Kong, Tao and Chu, Ruihang and Li, Yifeng and Wang, Peng and Li, Lei},
  booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={3571--3578},
  year={2021},
  organization={IEEE}
}
```

**[ICRA2021]** RGB Matters: Learning 7-DoF Grasp Poses on Monocular RGBD Images,       [[Paper](https://arxiv.org/pdf/2103.02184.pdf)].

*Keywords: 6D genral grasp; cluttered scene; RGB and single-view point cloud input; real-world dataset GraspNet1-billion.*

```latex
@inproceedings{gou2021RGB,
  title={RGB Matters: Learning 7-DoF Grasp Poses on Monocular RGBD Images},
  author={Minghao Gou, Hao-Shu Fang, Zhanda Zhu, Sheng Xu, Chenxi Wang, Cewu Lu},
  booktitle={Proceedings of the International Conference on Robotics and Automation (ICRA)},
  year={2021}
}
```

**[RSS2021]** Synergies Between Affordance and Geometry: 6-DoF Grasp Detection via Implicit Representations, [[Project](https://sites.google.com/view/rpl-giga2021)], [[Paper](https://arxiv.org/pdf/2104.01542.pdf)], [[Code](https://github.com/UT-Austin-RPL/GIGA)].

*Keywords: cluttered scene; 6D grasp; multi-task learning, 3D reconstruction and grasp detection; train the modelon self-supervised grasp trials data in simulation.*

```latex
@article{jiang2021synergies,
 author = {Jiang, Zhenyu and Zhu, Yifeng and Svetlik, Maxwell and Fang, Kuan and Zhu, Yuke},
 journal = {Robotics: science and systems},
 title = {Synergies Between Affordance and Geometry: 6-DoF Grasp Detection via Implicit Representations},
 year = {2021}
}
```

**[ICRA2021]** GPR: Grasp Pose Refinement Network for Cluttered Scenes,  [[Paper](https://arxiv.org/pdf/2105.08502.pdf)].

*Keywords: 6D grasp; two-stage method; self-made dataset in simulation; cluttered scene in simulation; PointNet++ backbone.*

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

*Keywords: 6D grasp; object grasp dataset is ACRONYM; cluttered scene in simulation; single-view scene point cloud(20000 points) input; backbone based on PointNet++.*

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

**[ICRA2021]** Robotic Grasping through Combined Image-Based Grasp Proposal and 3D Reconstruction, [[Paper](https://arxiv.org/pdf/2003.01649.pdf)].

*Keywords: input RGB-D image; single object; 6D grasp; multi-task learning, point cloud reconstruction and grasp generation.*

```latex
@INPROCEEDINGS{9562046,
  author={Yang, Daniel and Tosun, Tarik and Eisner, Benjamin and Isler, Volkan and Lee, Daniel},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={Robotic Grasping through Combined Image-Based Grasp Proposal and 3D Reconstruction}, 
  year={2021},
  volume={},
  number={},
  pages={6350-6356},
  doi={10.1109/ICRA48506.2021.9562046}}
```

**[ICRA2021]** REGNet: REgion-based Grasp Network for End-to-end Grasp Detection in Point Clouds, [[Paper](https://arxiv.org/pdf/2002.12647v1.pdf)].

*Keywords: input point cloud; PointNet++ backbone; 6D grasp; single pbject; 3-stage single-shot network, Score Network (SN), Grasp Region Network (GRN) and Refine Network (RN).*

```latex
@INPROCEEDINGS{9561920,
  author={Zhao, Binglei and Zhang, Hanbo and Lan, Xuguang and Wang, Haoyu and Tian, Zhiqiang and Zheng, Nanning},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={REGNet: REgion-based Grasp Network for End-to-end Grasp Detection in Point Clouds}, 
  year={2021},
  pages={13474-13480},
  doi={10.1109/ICRA48506.2021.9561920}}
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

**[CVPR2020]**  GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping,  [[Project&&Dataset](https://graspnet.net/index.html)],  [[Paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.pdf)],  [[Code](https://github.com/graspnet)], [[Supp](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Fang_GraspNet-1Billion_A_Large-Scale_CVPR_2020_supplemental.pdf)].

*Keywords: 6D general grasp; release a large-scale real-world dataset; cluttered scene; single-view scene point cloud input; PointNet++ backbone.*

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

*Keywords: 6D grasp; single object; single-view point cloud input; PointNet++ backbone; self-made synthetic dataset based on ShapeNetSem in simulation.*

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

**[CoRL2020]** Volumetric Grasping Network: Real-time 6 DOF Grasp Detection in Clutter, [[Paper](https://arxiv.org/pdf/2101.01132.pdf)], [[Code](https://github.com/ethz-asl/vgn)].

*Keywords: 6D grasp; Truncated Signed Distance Function (TSDF) representation of the scene; cluttered scene; trained on a synthetic grasping dataset generated with physics simulation.*

```latex
@inproceedings{breyer2020volumetric,
 title={Volumetric Grasping Network: Real-time 6 DOF Grasp Detection in Clutter},
 author={Breyer, Michel and Chung, Jen Jen and Ott, Lionel and Roland, Siegwart and Juan, Nieto},
 booktitle={Conference on Robot Learning},
 year={2020},
}
```

**[CoRL2020]** GDN: A Coarse-To-Fine (C2F) Representation for End-To-End 6-DoF Grasp Detection, [[Paper](https://proceedings.mlr.press/v155/jeng21a/jeng21a.pdf)].

*Keywords: 6D grasp; one-stage method; single object; PointNet++ backbone; self-made dataset based on YCB.*

```latex
@InProceedings{pmlr-v155-jeng21a,
  title =      {GDN: A Coarse-To-Fine (C2F) Representation for End-To-End 6-DoF Grasp Detection},
  author =       {Jeng, Kuang-Yu and Liu, Yueh-Cheng and Liu, Zhe Yu and Wang, Jen-Wei and Chang, Ya-Liang and Su, Hung-Ting and Hsu, Winston},
  booktitle =      {Proceedings of the 2020 Conference on Robot Learning},
  pages =      {220--231},
  year =      {2021},
  volume =      {155},
  series =      {Proceedings of Machine Learning Research},
  publisher =    {PMLR},
}
```

**[ICRA2020]** Learning to Generate 6-DoF Grasp Poses with Reachability Awareness, [[Paper](https://arxiv.org/pdf/1910.06404.pdf)].

*Keywords: 6D grasp; cluttered scene; sampling based grasp generation; point cloud voxelization, 3D CNN; grasp pose should be stable and reachable; train on synthetic dataset; self-supervised data collection.*

```latex
@INPROCEEDINGS{9197413,
  author={Lou, Xibai and Yang, Yang and Choi, Changhyun},
  booktitle={2020 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={Learning to Generate 6-DoF Grasp Poses with Reachability Awareness}, 
  year={2020},
  volume={},
  number={},
  pages={1532-1538},
  doi={10.1109/ICRA40945.2020.9197413}}
```

**[ICRA2020]** PointNet++ Grasping: Learning An End-to-end Spatial Grasp Generation Algorithm from Sparse Point Clouds, [[Paper](https://arxiv.org/ftp/arxiv/papers/2003/2003.09644.pdf)].

*Keywords: end-to-end approach, directly predict grasp; 6D grasp; PointNet++ backbone; single/multi-object scene; point cloud input.*

```latex
@inproceedings{ni2020pointnet++,
  title={Pointnet++ grasping: learning an end-to-end spatial grasp generation algorithm from sparse point clouds},
  author={Ni, Peiyuan and Zhang, Wenguang and Zhu, Xiaoxiao and Cao, Qixin},
  booktitle={2020 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={3619--3625},
  year={2020},
  organization={IEEE}
}
```

**[ICRA2020]** Real-Time, Highly Accurate Robotic Grasp Detection using Fully Convolutional Neural Network with Rotation Ensemble Module, [[Paper](https://arxiv.org/pdf/1812.07762.pdf)].

*Keywords: 2D grasp; RGB input; Cornell dataset.*

```latex
@inproceedings{park2020real,
  title={Real-time, highly accurate robotic grasp detection using fully convolutional neural network with rotation ensemble module},
  author={Park, Dongwon and Seo, Yonghyeok and Chun, Se Young},
  booktitle={2020 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={9397--9403},
  year={2020},
  organization={IEEE}
}
```

**[ICRA2020]** Action Image Representation: Learning Scalable Deep Grasping Policies with Zero Real World Data, [[Paper](https://arxiv.org/abs/2005.06594)].

*Keywords: 2D grasp; sampling-based grasp generation; a new grasp proposal representation.*

```latex
@INPROCEEDINGS{9197415,
  author={Khansari, Mohi and Kappler, Daniel and Luo, Jianlan and Bingham, Jeff and Kalakrishnan, Mrinal},
  booktitle={2020 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={Action Image Representation: Learning Scalable Deep Grasping Policies with Zero Real World Data}, 
  year={2020},
  pages={3597-3603},
  doi={10.1109/ICRA40945.2020.9197415}}
```

**[ICCV2019]** 6-DOF GraspNet: Variational Grasp Generation for Object Manipulation, [[Paper](https://arxiv.org/pdf/1905.10520.pdf)], [[Code](https://github.com/NVlabs/6dof-graspnet)].

*Keywords: 6D grasp; sampling then evaluation; PointNet++ backbone; generate dataset in simulation; single object point cloud and grasp input.*

```latex
@inproceedings{mousavian20196,
  title={6-dof graspnet: Variational grasp generation for object manipulation},
  author={Mousavian, Arsalan and Eppner, Clemens and Fox, Dieter},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={2901--2910},
  year={2019}
}
```

**[ICRA2019]** PointNetGPD: Detecting Grasp Configurations from Point Set, [[Paper](https://web.cs.ucla.edu/~xm/file/pointnetgpd_icra19.pdf)], [[Code](https://github.com/lianghongzhuo/PointNetGPD)].

*Keywords: 6D grasp; sampling then evaluation; single object cloud and grasp input; PointNet backbone; generate a large grasp dataset with YCB object set.*

```latex
@inproceedings{liang2019pointnetgpd,
  title={Pointnetgpd: Detecting grasp configurations from point sets},
  author={Liang, Hongzhuo and Ma, Xiaojian and Li, Shuang and G{\"o}rner, Michael and Tang, Song and Fang, Bin and Sun, Fuchun and Zhang, Jianwei},
  booktitle={2019 International Conference on Robotics and Automation (ICRA)},
  pages={3629--3635},
  year={2019},
  organization={IEEE}
}
```

### 3.2 Semantic grasping <span id="semantic-grasping"> </span>

**[RA-L2022]** REGRAD: A Large-Scale Relational Grasp Dataset for Safe and Object-Specific Robotic Grasping in Clutter, [[Paper](https://arxiv.org/pdf/2104.14118.pdf)], [[Code&&Dataset](https://github.com/poisonwine/REGRAD)].

*Keywords: release a dataset; cluttered scene; auto-generated in simulation; learn relationships among objetcs and grasps.*

```latex
@ARTICLE{9681218,
  author={Zhang, Hanbo and Yang, Deyu and Wang, Han and Zhao, Binglei and Lan, Xuguang and Ding, Jishiyu and Zheng, Nanning},
  journal={IEEE Robotics and Automation Letters}, 
  title={REGRAD: A Large-Scale Relational Grasp Dataset for Safe and Object-Specific Robotic Grasping in Clutter}, 
  year={2022},
  volume={7},
  number={2},
  pages={2929-2936},
  doi={10.1109/LRA.2022.3142401}}
```

**[RA-L2022]** Few-Shot Instance Grasping of Novel Objects in Clutter, [[Paper](https://ieeexplore.ieee.org/abstract/document/9773996)], [[Code](https://github.com/woundenfish/IGML)].

*Keywords: cluttered scene; grasp a specific object; meta-learning framework; 2D grasp.*

```latex
@ARTICLE{9773996,
  author={Guo, Weikun and Li, Wei and Hu, Ziye and Gan, Zhongxue},
  journal={IEEE Robotics and Automation Letters}, 
  title={Few-Shot Instance Grasping of Novel Objects in Clutter}, 
  year={2022},
  volume={7},
  number={3},
  pages={6566-6573},
  doi={10.1109/LRA.2022.3174648}}
```

**[ICRA2022]** Learning Object Relations with Graph Neural Networks for Target-Driven Grasping in Dense Clutter, [[Project](https://sites.google.com/umn.edu/graph-grasping)], [[Paper](https://arxiv.org/pdf/2203.00875.pdf)].

*Keywords: target-driven grasp; cluttered scene; 6-D grasp; sampling based grasp generation; shape completion-assisted grasp sampling; formulate grasp graph, nodes representing object, edges indicating spatial relations between the objects; train on synthetic dataset; input scene RGB and query image.*

- [ ] Bibtex to be update(arxiv to icra)

```latex
@article{lou2022learning,
  title={Learning Object Relations with Graph Neural Networks for Target-Driven Grasping in Dense Clutter},
  author={Lou, Xibai and Yang, Yang and Choi, Changhyun},
  journal={arXiv preprint arXiv:2203.00875},
  year={2022}
}
```

**[ICRA2022]** Interactive Robotic Grasping with Attribute-Guided Disambiguation,  [[Project](https://sites.google.com/umn.edu/attr-disam)], [[Paper](https://arxiv.org/pdf/2203.08037.pdf)].

*Keywords: cluttered scene; input scene RGBD and query language; 6D grasp; vision-and-language grounding module predicts target scores and attribute scores; attribute-guided  partially observable Markov decision process for language disambiguation(ask questions).*

- [ ] Bibtex to be updated(arxiv to icra)

```latex
@article{yang2022interactive,
  title={Interactive Robotic Grasping with Attribute-Guided Disambiguation},
  author={Yang, Yang and Lou, Xibai and Choi, Changhyun},
  journal={arXiv preprint arXiv:2203.08037},
  year={2022}
}
```

**[ICRA2022]** I Know What You Draw: Learning Grasp Detection Conditioned on a Few Freehand Sketches, [[Project](https://hetolin.github.io/Skt_grasp/)], [[Paper](https://arxiv.org/pdf/2205.04026.pdf)].

*Keywords: 2D planar grasp; cluttered scene; target grasps by understanding freehand sketches; RGB image and graph-represented sketch input.*

- [ ] Bibtex to be update(arxiv to icra)

```latex
@article{lin2022know,
  title={I Know What You Draw: Learning Grasp Detection Conditioned on a Few Freehand Sketches},
  author={Lin, Haitao and Cheang, Chilam and Fu, Yanwei and Xue, Xiangyang},
  journal={arXiv preprint arXiv:2205.04026},
  year={2022}
}
```

**[ICRA2022]** Learning 6-DoF Object Poses to Grasp Category-level Objects by Language Instructions, [[Project](https://baboon527.github.io/lang_6d/)], [[Paper](https://arxiv.org/pdf/2205.04028v1.pdf)], [[Code](https://github.com/baboon527/lang_6d)].

*Keywords: grasp target object based on language description; two-stage method; 2D visual grounding, category-level object pose estimation; RGBD and language description input.*

- [ ] Bibtex to be update(arxiv to icra)

```latex
@article{cheang2022learning,
  title={Learning 6-DoF Object Poses to Grasp Category-level Objects by Language Instructions},
  author={Cheang, Chilam and Lin, Haitao and Fu, Yanwei and Xue, Xiangyang},
  journal={arXiv preprint arXiv:2205.04028},
  year={2022}
}
```

**[ICRA2022]** CaTGrasp: Learning Category-Level Task-Relevant Grasping in Clutter from Simulation,  [[Project](https://sites.google.com/view/catgrasp)],  [[Paper](https://arxiv.org/pdf/2109.09163v1.pdf)],  [[Code](https://github.com/wenbowen123/catgrasp)].  

*Keywords: 6D category-level task-oriented grasp; cluttered scene in simulation;  self-supervised in simulation.*

- [ ] Bibtex to be updated(arxiv to icra)

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

**[ICRA2021]** A Joint Network for Grasp Detection Conditioned on Natural Language Commands, [[Paper](https://arxiv.org/pdf/2104.00492.pdf)].

*Keywords: 2D planar grasp; structured language command and rgb input; VMRD dataset; target-sepcific grasp.*

```latex
@inproceedings{chen2021joint,
  title={A joint network for grasp detection conditioned on natural language commands},
  author={Chen, Yiye and Xu, Ruinian and Lin, Yunzhi and Vela, Patricio A},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={4576--4582},
  year={2021},
  organization={IEEE}
}
```

**[ICRA2021]** End-to-end Trainable Deep Neural Network for Robotic Grasp Detection and Semantic Segmentation from RGB, [[Paper](https://arxiv.org/pdf/2107.05287)].

*Keywords: 2D grasp; joint grasp detection and semantic segmentation.*

```latex
@INPROCEEDINGS{9561398,
  author={Ainetter, Stefan and Fraundorfer, Friedrich},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={End-to-end Trainable Deep Neural Network for Robotic Grasp Detection and Semantic Segmentation from RGB}, 
  year={2021},
  pages={13452-13458},
  doi={10.1109/ICRA48506.2021.9561398}}
```

**[RSS2021]** INVIGORATE: Interactive Visual Grounding and Grasping in Clutter, [[Paper](http://roboticsproceedings.org/rss17/p020.pdf)].

*Keywords: input language expressions and RGB; cluttered scene; train separate neural networks for object detection, for visual grounding, for question generation, and for object blocking relationships detection and grasping.*

```latex
@INPROCEEDINGS{ZhangLu-RSS-21, 
    AUTHOR    = {Hanbo Zhang AND Yunfan Lu AND Cunjun Yu AND David Hsu AND Xuguang Lan AND Nanning Zheng}, 
    TITLE     = {{INVIGORATE: Interactive Visual Grounding and Grasping in Clutter}}, 
    BOOKTITLE = {Proceedings of Robotics: Science and Systems}, 
    YEAR      = {2021}, 
    ADDRESS   = {Virtual}, 
    MONTH     = {July}, 
    DOI       = {10.15607/RSS.2021.XVII.020} 
} 
```

**[ICRA2020]** 6-DOF Grasping for Target-driven Object Manipulation in Clutter, [[Paper](https://arxiv.org/pdf/1912.03628.pdf)].

*Keywords: 6D grasp; cluttered scene; grasp target object; RGB-D input; sampling based grasp.*

```latex
@INPROCEEDINGS{9197318,
  author={Murali, Adithyavairavan and Mousavian, Arsalan and Eppner, Clemens and Paxton, Chris and Fox, Dieter},
  booktitle={2020 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={6-DOF Grasping for Target-driven Object Manipulation in Clutter}, 
  year={2020},
  pages={6232-6238},
  doi={10.1109/ICRA40945.2020.9197318}}
```

**[CoRL2020]** Same Object, Different Grasps: Data and SemanticKnowledge for Task-Oriented Grasping, [[Project](https://sites.google.com/view/taskgrasp)], [[Paper](https://proceedings.mlr.press/v155/murali21a/murali21a.pdf)], [[Code](https://github.com/adithyamurali/TaskGrasp)], [[Dataset](https://drive.google.com/file/d/1aZ0k43fBIZZQPPPraV-z6itpqCHuDiUU/view)].

*Keywords: 6D task-oriented grasp; single object; real-world data; object point cloud and goal task input; PointNet++ backbone for point cloud,  Graph Convolutional Network for object and task semantic knowledge.*

```latex
@inproceedings{murali2020taskgrasp,
  title={Same Object, Different Grasps: Data and Semantic Knowledge for Task-Oriented Grasping},
  author={Murali, Adithyavairavan and Liu, Weiyu and Marino, Kenneth and Chernova, Sonia and Gupta, Abhinav},
  booktitle={Conference on Robot Learning},
  year={2020}
}
```

**[IJRR2020]** Learning Task-Oriented Grasping for Tool Manipulation from Simulated Self-Supervision, [[Project](https://sites.google.com/view/task-oriented-grasp/)], [[Paper](https://arxiv.org/pdf/1806.09266.pdf)].

*Keywords: task-oriented grasping; manipulation policy; self-supervised in simulation; single object; planar grasp; depth image input; 2 tasks.*

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

**[ICRA2020]** CAGE: Context-Aware Grasping Engine, [[Paper](https://arxiv.org/pdf/1909.11142.pdf)], [[Code](https://github.com/wliu88/rail_semantic_grasping)].

*Keywords: single object; semantic context including task, object state, object material, object affordance; semantic context and sampled grasps input; output ranking of grasps ordered by their suitability to the context.*

```latex
@inproceedings{liu2020cage,
  title={Cage: Context-aware grasping engine},
  author={Liu, Weiyu and Daruna, Angel and Chernova, Sonia},
  booktitle={2020 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={2550--2556},
  year={2020},
  organization={IEEE}
}
```

**[IROS2019]** Task-oriented Grasping in Object Stacking Scenes with CRF-based Semantic Model, [[Paper](https://ieeexplore.ieee.org/document/8967992)].

*Keywords: task-oriented grasping; cluttered scene in simulation; planar grasp; 11 tasks, 10 object categories, 100 objects; depth image input.*

```latex
@inproceedings{yang2019task,
  title={Task-oriented grasping in object stacking scenes with crf-based semantic model},
  author={Yang, Chenjie and Lan, Xuguang and Zhang, Hanbo and Zheng, Nanning},
  booktitle={2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={6427--6434},
  year={2019},
  organization={IEEE}
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

*Keywords: single object point cloud and task name input; output affordance detection, no grasp; 5 tasks, 10 object classes; generalize to novel object class.*

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

## 4. Research Groups  <span id="research-groups"> </span>

- [SJTU Machine Vision and Intelligence Group](https://mvig.sjtu.edu.cn/),  Prof. Cewu Lu [[Google Scholar](https://scholar.google.de/citations?user=QZVQEWAAAAAJ&hl=zh-CN&oi=ao)].
- [XJTU, College of Artificial Intelligence](http://www.iair.xjtu.edu.cn/), Prof. Xuguang Lan [[Homepage](https://gr.xjtu.edu.cn/en/web/zeuslan/information)].
- [UT Austin Robot Perception and Learning Lab](https://rpl.cs.utexas.edu/), Prof. Yuke Zhu [[Google Scholar](https://scholar.google.com/citations?user=mWGyYMsAAAAJ&hl=en)].
