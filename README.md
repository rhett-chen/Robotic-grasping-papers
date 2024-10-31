# Robotic Grasping Papers and Codes

This repo is a paper list of Robotic-Grasping and some related tasks (6D pose estimation, visual grounding, robotic manipulation, etc).

Abbreviation:  
* **ICRA** is IEEE International Conference on Robotics and Automation; 
* **CVPR** is IEEE Conference on Computer Vision and Pattern Recognition;  
* **ICCV** is IEEE International Conference on Computer Vision; 
* **ECCV** is European Conference on Computer Vision; 
* **CoRL** is Conference on Robot Learning; 
* **NIPS** is Conference on Neural Information Processing Systems;  
* **RA-L** is IEEE Robotics and Automation Letters;
* **Humanoids** is IEEE-RAS International Conference on Humanoid Robots; 
* **IJRR** is The International Journal of Robotics Research; 
* **IROS** is IEEE/RSJ International Conference on Intelligent Robots and Systems; 
* **ACM MM** is  ACM International Conference on Multimedia; 
* **RSS** is Robotics: Science and Systems;
* **T-RO** is IEEE Transactions on Robotics.

- [Robotic Grasping Papers and Codes](#robotic-grasping-papers-and-codes)
  - [1. Survey Papers](#1-survey-papers)
  - [2. Related Tasks](#2-related-tasks)
    - [2.1 Visual grounding](#21-visual-grounding)
    - [2.2 Robotic manipulation](#22-robotic-manipulation)
    - [2.3 6D pose estimation](#23-6d-pose-estimation)
    - [2.4 Datasets](#24-datasets)
  - [3. Grasp Detection](#3-grasp-detection)
    - [3.1 General grasping](#31-general-grasping)
    - [3.2 Dexterous grasping](#32-dexterous-grasping)
    - [3.3 Semantic grasping](#33-semantic-grasping)
    - [3.4 Dynamic Grasping](#34-dynamic-grasping)
  - [4. Research Groups](#4-research-groups)

## 1. Survey Papers

**[T-RO2023]** Deep Learning Approaches to Grasp Synthesis: A Review,  [[Project](https://rhys-newbury.github.io/projects/6dof/)], [[Paper](https://arxiv.org/pdf/2207.02556.pdf)].

*Keywords: focus on 6D grasping; sampling based approaches, direct regression, using shape-completion, reinforcement learning or considering semantics.*

```latex
@ARTICLE{10149823,
  author={Newbury, Rhys and Gu, Morris and Chumbley, Lachlan and Mousavian, Arsalan and Eppner, Clemens and Leitner, Jürgen and Bohg, Jeannette and Morales, Antonio and Asfour, Tamim and Kragic, Danica and Fox, Dieter and Cosgun, Akansel},
  journal={IEEE Transactions on Robotics}, 
  title={Deep Learning Approaches to Grasp Synthesis: A Review}, 
  year={2023},
  volume={39},
  number={5},
  pages={3994-4015},
  doi={10.1109/TRO.2023.3280597}
}

```

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

## 2. Related Tasks 
### 2.1 Visual grounding 

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

**[CVPR2022]** Text2Pos: Text-to-Point-Cloud Cross-Modal Localization, [[Project](https://text2pos.github.io/)], [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Kolmet_Text2Pos_Text-to-Point-Cloud_Cross-Modal_Localization_CVPR_2022_paper.pdf)], [[Code](https://github.com/mako443/Text2Pos-CVPR2022)].

*Keywords: city-scale outdoor point cloud localizaiotn; provide KITTI360Pose dataset based on KITTI360; coarse-to-fine method, first retrieval sub-regions, then refine the position using matching-based fine localization module.*

```latex
@inproceedings{dendorfer21iccv, 
  title = {Text2Pos: Text-to-Point-Cloud Cross-Modal Localization}, 
  author={Manuel Kolmet and Qunjie Zhou and Aljosa Osep and Laura Leal-Taix{'e}}, 
  booktitle = { IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  year = {2022}, 
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

**[CVPR2022]** Improving Visual Grounding with Visual-Linguistic Verification and Iterative Reasoning, [[Paper](https://arxiv.org/pdf/2205.00272.pdf)], [[Code](https://github.com/yangli18/VLTVG)].

*Keywords: 2D visual grounding; one-stage method; transformer-based architecture.*

```latex
@inproceedings{yang2022vgvl,
  title={Improving Visual Grounding with Visual-Linguistic Verification and Iterative Reasoning},
  author={Yang, Li and Xu, Yan and Yuan, Chunfeng and Liu, Wei and Li, Bing and Hu, Weiming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

**[arXiv2021]** Looking Outside the Box to Ground Language in 3D Scenes, [[Paper](https://arxiv.org/pdf/2112.08879.pdf)], [[Code](https://github.com/nickgkan/beauty_detr)].

*Keywords: propose BEAUTY-DETR, a transformer like architecture for 3D visual grounding; input scene point cloud, query text and object proposals generated by pretrained detector.*

```latex
@article{jain2021looking,
  title={Looking Outside the Box to Ground Language in 3D Scenes},
  author={Jain, Ayush and Gkanatsios, Nikolaos and Mediratta, Ishita and Fragkiadaki, Katerina},
  journal={arXiv preprint arXiv:2112.08879},
  year={2021}
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

### 2.2 Robotic manipulation

**[RSS2024]** Any-point Trajectory Modeling for Policy Learning, [[Project](https://xingyu-lin.github.io/atm/)], [[Paper](https://arxiv.org/pdf/2401.00025)], [[Code](https://github.com/Large-Trajectory-Model/ATM)].

*Keywords: represent each state in a video as a set of points to model the transition dynamics, provide detailed control guidance to enable the learning of robust visuomotor policies with minimal action-labeled data; two stage, first, train a track transformer to predict future point trajectories given the current image, language instruction, initial positions of sampled points, second, learn a track-guided policy to predict the control actions;*

```latex
@article{wen2023atm,
  title={Any-point trajectory modeling for policy learning},
  author={Wen, Chuan and Lin, Xingyu and So, John and Chen, Kai and Dou, Qi and Gao, Yang and Abbeel, Pieter},
  journal={arXiv preprint arXiv:2401.00025},
  year={2023}
}
```

**[ICLR2024]** Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation, [[Project](https://gr1-manipulation.github.io/)], [[Paper](https://arxiv.org/pdf/2312.13139)], [[Code](https://github.com/bytedance/GR-1)].

*Keywords: leverage large-scale video generative pre-training for manipulation, finetune on robotic manipulation data; propose GR-1, takes as inputs a language instruction, a sequence of observation images, and a sequence of robot states, predicts robot actions (end-effector pose and gripper width) as well as future images in an end-to-end manner.*

```latex
@inproceedings{wu2023unleashing,
  title={Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation},
  author={Wu, Hongtao and Jing, Ya and Cheang, Chilam and Chen, Guangzeng and Xu, Jiafeng and Li, Xinghang and Liu, Minghuan and Li, Hang and Kong, Tao},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```

**[arXiv2024]** Expressive Whole-Body Control for Humanoid Robots, [[Project](https://expressive-humanoid.github.io/)], [[Paper](https://arxiv.org/pdf/2402.16796v1)].

*Keywords: Unitree H1 humanoid robot, 19-DoF; whole-body control; key idea is to let the upper body imitate human motions and the relax the motion imitation of two legs; train goal-conditioned RL model in Iassc Gym simulator.*

```latex
@article{cheng2024express,
  title={Expressive Whole-Body Control for Humanoid Robots},
  author={Cheng, Xuxin and Ji, Yandong and Chen, Junming and Yang, Ruihan and Yang, Ge and Wang, Xiaolong},
  journal={arXiv preprint arXiv:2402.16796},
  year={2024}
}
```

**[ICRA2024]** Robot Synesthesia: In-Hand Manipulation with Visuotactile Sensing, [[Project](https://yingyuan0414.github.io/visuotactile/)], [[Paper](https://arxiv.org/pdf/2312.01853v2)].

*Keywords: In-hand object rotation task, one or two objects; visual and tactile infomation; represent tactile data as point cloud to integrate with visual data; train teacher policy using RL in simulator with oracle state information, and then distill to a student policy to deploy on real robot.*

```latex
@article{yuan2023robot,
  title={Robot synesthesia: In-hand manipulation with visuotactile sensing},
  author={Yuan, Ying and Che, Haichuan and Qin, Yuzhe and Huang, Binghao and Yin, Zhao-Heng and Lee, Kang-Won and Wu, Yi and Lim, Soo-Chul and Wang, Xiaolong},
  journal={arXiv preprint arXiv:2312.01853},
  year={2023}
}
```

**[IROS2023]** MOMA-Force: Visual-Force Imitation for Real-World Mobile Manipulation, [[Paper](https://arxiv.org/pdf/2308.03624)].

*Keywords: contact-rich manipulation tasks; propose a visual-force imitation method, combine representation learning for perception, imitation learning for motion generation, and admittance whole-body control.*

```latex
@inproceedings{yang2023visualforce
      author = {Yang, Taozheng and Jing, Ya and Wu, Hongtao and Xu, Jiafeng and Sima, Kuankuan and Chen, Guangzeng and Sima, Qie and Kong, Tao},
      title = {MOMA-Force: Visual-Force Imitation for Real-World Mobile Manipulation},
      booktitle = {2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
      year = {2023}
}
```

**[CVPR2023]** Learning Human-to-Robot Handovers from Point Clouds, [[Project](https://handover-sim2real.github.io/)], [[Paper](https://arxiv.org/pdf/2303.17592.pdf)], [[Code](https://handover-sim2real.github.io/#)].

*Keywords: point cloud input;  6D grasp; trained by interacting with the humans in simulation environment; reinforcement learning, two-stage training scheme.*

*Motivation: To close the gap of human-in-the-loop policy training for human-to-robot handover, this paper introduces a vision-based learning framework for H2R handovers that is trained with a human-in-the-loop.*

```latex
@inproceedings{christen2023handoversim2real,
      title = {Learning Human-to-Robot Handovers from Point Clouds},
      author = {Christen, Sammy and Yang, Wei and Pérez-D'Arpino, Claudia and Hilliges, Otmar and Fox, Dieter and Chao, Yu-Wei},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2023}
}
```

**[CoRL2022]** BEHAVIOR-1K: A Benchmark for Embodied AI with 1,000 Everyday Activities and Realistic Simulation, [[Project](https://behavior.stanford.edu/behavior-1k)], [[Paper](https://proceedings.mlr.press/v205/li23a/li23a.pdf)].

*Keywords: a comprehensive simulation benchmark for human-centered robotic; 1000 everyday activities, 50 scenes, 5000+ objects.*

```
@InProceedings{pmlr-v205-li23a,
  title = 	 {BEHAVIOR-1K: A Benchmark for Embodied AI with 1,000 Everyday Activities and Realistic Simulation},
  author =       {Li, Chengshu and Zhang, Ruohan and Wong, Josiah and Gokmen, Cem and Srivastava, Sanjana and Mart\'in-Mart\'in, Roberto and Wang, Chen and Levine, Gabrael and Lingelbach, Michael and Sun, Jiankai and Anvari, Mona and Hwang, Minjune and Sharma, Manasi and Aydin, Arman and Bansal, Dhruva and Hunter, Samuel and Kim, Kyu-Young and Lou, Alan and Matthews, Caleb R and Villa-Renteria, Ivan and Tang, Jerry Huayang and Tang, Claire and Xia, Fei and Savarese, Silvio and Gweon, Hyowon and Liu, Karen and Wu, Jiajun and Fei-Fei, Li},
  booktitle = 	 {Proceedings of The 6th Conference on Robot Learning},
  pages = 	 {80--93},
  year = 	 {2023},
  volume = 	 {205},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {14--18 Dec},
  publisher =    {PMLR},
}

```

**[ECCV2022]** DexMV: Imitation Learning for Dexterous Manipulation from Human Videos, [[Project](https://yzqin.github.io/dexmv/)], [[Paper](https://arxiv.org/pdf/2108.05877.pdf)], [[Code](https://github.com/yzqin/dexmv-sim)].

*Keywords: dexterous manipulation; record large-scale demonstrations of human hand conducting same tasks, and convert human motion to robot demonstrations; train imitation learning agent in simulation environemnt; benchmark multiple imitation learning algorithms with the collected demonstrations.*

*Motivation: To tackle complex robot dexterous manipulation tasks by imitation learning.*

```latex
@inproceedings{qin2022dexmv,
  title={Dexmv: Imitation learning for dexterous manipulation from human videos},
  author={Qin, Yuzhe and Wu, Yueh-Hua and Liu, Shaowei and Jiang, Hanwen and Yang, Ruihan and Fu, Yang and Wang, Xiaolong},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part XXXIX},
  pages={570--587},
  year={2022},
  organization={Springer}
}
```

**[CVPR2022]** IFOR: Iterative Flow Minimization for Robotic Object Rearrangement, [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Goyal_IFOR_Iterative_Flow_Minimization_for_Robotic_Object_Rearrangement_CVPR_2022_paper.pdf)].

*Keywords: input RGBD image of the original and final scenes; object rearrangement for unknown objects, handle objects with translation and planar rotations; trained on synthetic data, transfer to real-world in zero-shot manner.*

```latex
@InProceedings{Goyal_2022_CVPR,
    author    = {Goyal, Ankit and Mousavian, Arsalan and Paxton, Chris and Chao, Yu-Wei and Okorn, Brian and Deng, Jia and Fox, Dieter},
    title     = {IFOR: Iterative Flow Minimization for Robotic Object Rearrangement},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {14787-14797}
}
```

**[RA-L2022]** CALVIN: A Benchmark for Language-conditioned Policy Learning for Long-horizon Robot Manipulation Tasks, [[Paper](https://arxiv.org/pdf/2112.03227.pdf)], [[Code](https://github.com/mees/calvin)].

*Keywords: language conditioned long-horizon manipulation; 34 tasks; 4 simulation environments; 7-Dof Panda robot; a static camera and a robot gripper camera; RGB-D image; unstructured demonstrations datasets, ∼2.4M interaction steps.*

```latex
@ARTICLE{9788026,
  author={Mees, Oier and Hermann, Lukas and Rosete-Beas, Erick and Burgard, Wolfram},
  journal={IEEE Robotics and Automation Letters}, 
  title={CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks}, 
  year={2022},
  volume={7},
  number={3},
  pages={7327-7334},
  doi={10.1109/LRA.2022.3180108}
}
```

**[RSS2021]** NeRP: Neural Rearrangement Planning for Unknown Objects, [[Paper](http://www.roboticsproceedings.org/rss17/p072.pdf)].

*Keywords: multi-step object rearrangement planning, for unknown objects; input RGBD image of the original and final scenes; need to segment out unique objects in scene, compute object alignment between current and goal state; train on synthetic data.*

```latex
@INPROCEEDINGS{Qureshi-RSS-21, 
    AUTHOR    = {Ahmed H Qureshi AND Arsalan Mousavian AND Chris Paxton AND Michael Yip AND Dieter Fox}, 
    TITLE     = {NeRP: Neural Rearrangement Planning for Unknown Objects}, 
    BOOKTITLE = {Proceedings of Robotics: Science and Systems}, 
    YEAR      = {2021}, 
    ADDRESS   = {Virtual}, 
    MONTH     = {July}, 
    DOI       = {10.15607/RSS.2021.XVII.072} 
} 
```

**[CoRL2021]** CLIPort: What and Where Pathways for Robotic Manipulation, [[Project](https://cliport.github.io/)], [[Paper](https://arxiv.org/pdf/2109.12098.pdf)], [[Code](https://github.com/cliport/cliport)].

*Keywords: propose a two-stream architecture with semantic and spatial pathways for vision-based manipulation; propose CLIPORT, a language-conditioned imitation learning agent, can learn a single language-conditioned policy for various tabletop tasks.*

```latex
@inproceedings{shridhar2021cliport,
  title     = {CLIPort: What and Where Pathways for Robotic Manipulation},
  author    = {Shridhar, Mohit and Manuelli, Lucas and Fox, Dieter},
  booktitle = {Proceedings of the 5th Conference on Robot Learning (CoRL)},
  year      = {2021},
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

**[ICRA2022]** Audio-Visual Grounding Referring Expression for Robotic Manipulation, [[Paper](https://arxiv.org/pdf/2109.10571.pdf)].

*Keywords: a novel task, audio-visual grounding referring expression for robotic manipulation; establishe a dataset which contains visual data, auditory data and manipulation instructions.*

```latex
@INPROCEEDINGS{9811895,
  author={Wang, Yefei and Wang, Kaili and Wang, Yi and Guo, Di and Liu, Huaping and Sun, Fuchun},
  booktitle={2022 International Conference on Robotics and Automation (ICRA)}, 
  title={Audio-Visual Grounding Referring Expression for Robotic Manipulation}, 
  year={2022},
  pages={9258-9264},
  doi={10.1109/ICRA46639.2022.9811895}
}
```

**[ICRA2022]** StructFormer: Learning Spatial Structurefor Language-Guided Semantic Rearrangement of Novel Objects, [[Paper](https://arxiv.org/pdf/2110.10189.pdf)].

*Keywords: language-guided semantic rearrangement; transformer-based method; scene point cloud and structured language command input; output plan sequence, no 6D grasp.*

```latex
@INPROCEEDINGS{9811931,
  author={Liu, Weiyu and Paxton, Chris and Hermans, Tucker and Fox, Dieter},
  booktitle={2022 International Conference on Robotics and Automation (ICRA)}, 
  title={StructFormer: Learning Spatial Structure for Language-Guided Semantic Rearrangement of Novel Objects}, 
  year={2022},
  pages={6322-6329},
  doi={10.1109/ICRA46639.2022.9811931}
}
```

### 2.3 6D pose estimation

**[CVPR2022]** OnePose: One-Shot Object Pose Estimation without CAD Models, [[Project](https://zju3dv.github.io/onepose/)], [[Paper](https://arxiv.org/pdf/2205.12257.pdf)], [[Code](https://github.com/zju3dv/OnePose)], [[Dataset](https://zjueducn-my.sharepoint.com/personal/zihaowang_zju_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fzihaowang%5Fzju%5Fedu%5Fcn%2FDocuments%2Fonepose%5Fdataset%5Frelease&ga=1)].

*Keywords: handle objects in arbitrary categories without instance or category-specific network training; release a large-scale dataset; input RGB video scan of the object and query image.*

*Motivation: To alleviate the demand for CAD models or category-specific training.*

```latex
@article{sun2022onepose,
    title={{OnePose}: One-Shot Object Pose Estimation without {CAD} Models},
    author = {Sun, Jiaming and Wang, Zihao and Zhang, Siyu and He, Xingyi and Zhao, Hongcheng and Zhang, Guofeng and Zhou, Xiaowei},
    journal={CVPR},
    year={2022},
}
```

**[CVPR2022]** CPPF: Towards Robust Category-Level 9D Pose Estimation in the Wild, [[Paper](https://arxiv.org/pdf/2203.03089.pdf)], [[Code](https://github.com/qq456cvb/CPPF)].

*Keywords: category-level; point-pair features; voting method; sim-to-real transfer, trained on synthetic models, tested on real-world data, need an instance segmentation network.*

```latex
@inproceedings{you2022cppf,
  title={CPPF: Towards Robust Category-Level 9D Pose Estimation in the Wild},
  author={You, Yang and Shi, Ruoxi and Wang, Weiming and Lu, Cewu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

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

**[ICRA2019]** Making Sense of Vision and Touch: Self-Supervised Learning of Multimodal Representations for Contact-Rich Tasks, [[Project](https://sites.google.com/view/visionandtouch)], [[Paper](https://www.cs.utexas.edu/~yukez/publications/papers/lee-icra19-making.pdf)], [[Code](https://github.com/stanford-iprl-lab/multimodal_representation)].

*Keywords: **ICRA2019 best paper award**; multimodal representation learning for contact rich tasks; self-supervised representation learning; decouple representation learning and policy learning, so it can achieve practical sample efficiency on real robot.*

```latex
@INPROCEEDINGS{8793485,
  author={Lee, Michelle A. and Zhu, Yuke and Srinivasan, Krishnan and Shah, Parth and Savarese, Silvio and Fei-Fei, Li and Garg, Animesh and Bohg, Jeannette},
  booktitle={2019 International Conference on Robotics and Automation (ICRA)}, 
  title={Making Sense of Vision and Touch: Self-Supervised Learning of Multimodal Representations for Contact-Rich Tasks}, 
  year={2019},
  pages={8943-8950},
  doi={10.1109/ICRA.2019.8793485}
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

### 2.4 Datasets

**[arXiv2023]** Open X-Embodiment: Robotic Learning Datasets and RT-X Models, [[Project](https://robotics-transformer-x.github.io/)], [[Paper](https://robotics-transformer-x.github.io/paper.pdf)], [[Dataset](https://github.com/google-deepmind/open_x_embodiment)].

*Keywords: assemble a dataset from 22 different robots, demonstrating 527 skills (160266 tasks); learn generalizable robot policies.*

```latex
@misc{open_x_embodiment_rt_x_2023,
title={Open {X-E}mbodiment: Robotic Learning Datasets and {RT-X} Models},
author = {Open X-Embodiment Collaboration and Abhishek Padalkar and Acorn Pooley and Ajinkya Jain and Alex Bewley and Alex Herzog and Alex Irpan and Alexander Khazatsky and Anant Rai and Anikait Singh and Anthony Brohan and Antonin Raffin and Ayzaan Wahid and Ben Burgess-Limerick and Beomjoon Kim and Bernhard Schölkopf and Brian Ichter and Cewu Lu and Charles Xu and Chelsea Finn and Chenfeng Xu and Cheng Chi and Chenguang Huang and Christine Chan and Chuer Pan and Chuyuan Fu and Coline Devin and Danny Driess and Deepak Pathak and Dhruv Shah and Dieter Büchler and Dmitry Kalashnikov and Dorsa Sadigh and Edward Johns and Federico Ceola and Fei Xia and Freek Stulp and Gaoyue Zhou and Gaurav S. Sukhatme and Gautam Salhotra and Ge Yan and Giulio Schiavi and Hao Su and Hao-Shu Fang and Haochen Shi and Heni Ben Amor and Henrik I Christensen and Hiroki Furuta and Homer Walke and Hongjie Fang and Igor Mordatch and Ilija Radosavovic and Isabel Leal and Jacky Liang and Jaehyung Kim and Jan Schneider and Jasmine Hsu and Jeannette Bohg and Jeffrey Bingham and Jiajun Wu and Jialin Wu and Jianlan Luo and Jiayuan Gu and Jie Tan and Jihoon Oh and Jitendra Malik and Jonathan Tompson and Jonathan Yang and Joseph J. Lim and João Silvério and Junhyek Han and Kanishka Rao and Karl Pertsch and Karol Hausman and Keegan Go and Keerthana Gopalakrishnan and Ken Goldberg and Kendra Byrne and Kenneth Oslund and Kento Kawaharazuka and Kevin Zhang and Keyvan Majd and Krishan Rana and Krishnan Srinivasan and Lawrence Yunliang Chen and Lerrel Pinto and Liam Tan and Lionel Ott and Lisa Lee and Masayoshi Tomizuka and Maximilian Du and Michael Ahn and Mingtong Zhang and Mingyu Ding and Mohan Kumar Srirama and Mohit Sharma and Moo Jin Kim and Naoaki Kanazawa and Nicklas Hansen and Nicolas Heess and Nikhil J Joshi and Niko Suenderhauf and Norman Di Palo and Nur Muhammad Mahi Shafiullah and Oier Mees and Oliver Kroemer and Pannag R Sanketi and Paul Wohlhart and Peng Xu and Pierre Sermanet and Priya Sundaresan and Quan Vuong and Rafael Rafailov and Ran Tian and Ria Doshi and Roberto Martín-Martín and Russell Mendonca and Rutav Shah and Ryan Hoque and Ryan Julian and Samuel Bustamante and Sean Kirmani and Sergey Levine and Sherry Moore and Shikhar Bahl and Shivin Dass and Shuran Song and Sichun Xu and Siddhant Haldar and Simeon Adebola and Simon Guist and Soroush Nasiriany and Stefan Schaal and Stefan Welker and Stephen Tian and Sudeep Dasari and Suneel Belkhale and Takayuki Osa and Tatsuya Harada and Tatsuya Matsushima and Ted Xiao and Tianhe Yu and Tianli Ding and Todor Davchev and Tony Z. Zhao and Travis Armstrong and Trevor Darrell and Vidhi Jain and Vincent Vanhoucke and Wei Zhan and Wenxuan Zhou and Wolfram Burgard and Xi Chen and Xiaolong Wang and Xinghao Zhu and Xuanlin Li and Yao Lu and Yevgen Chebotar and Yifan Zhou and Yifeng Zhu and Ying Xu and Yixuan Wang and Yonatan Bisk and Yoonyoung Cho and Youngwoon Lee and Yuchen Cui and Yueh-hua Wu and Yujin Tang and Yuke Zhu and Yunzhu Li and Yusuke Iwasawa and Yutaka Matsuo and Zhuo Xu and Zichen Jeff Cui},
howpublished  = {\url{https://arxiv.org/abs/2310.08864}},
year = {2023},
}
```

**[CVPR2023]** Habitat-Matterport 3D Semantics Dataset, [[Project](https://aihabitat.org/datasets/hm3d-semantics/)], [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Yadav_Habitat-Matterport_3D_Semantics_Dataset_CVPR_2023_paper.pdf)], [[Code](https://github.com/matterport/habitat-matterport-3dresearch)].

*Keywords: the largest dataset of 3D real-world spaces with densely annotated semantics; 142646 objects; 216 3D spaces; 3100 rooms;*

```latex
@InProceedings{Yadav_2023_CVPR,
    author    = {Yadav, Karmesh and Ramrakhya, Ram and Ramakrishnan, Santhosh Kumar and Gervet, Theo and Turner, John and Gokaslan, Aaron and Maestre, Noah and Chang, Angel Xuan and Batra, Dhruv and Savva, Manolis and Clegg, Alexander William and Chaplot, Devendra Singh},
    title     = {Habitat-Matterport 3D Semantics Dataset},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {4927-4936}
}
```

**[CVPR2023]** OmniObject3D: Large-Vocabulary 3D Object Dataset for Realistic Perception, Reconstruction and Generation，[[Project](https://omniobject3d.github.io/)], [[Paper](https://arxiv.org/pdf/2301.07525.pdf)], [[Code](https://github.com/omniobject3d/OmniObject3D)].

*Keywords: 6000 scanned objects, 190 categories; Each 3D object is captured with both 2D and 3D sensors, providing textured meshes, point clouds, multi-view rendered images, and multiple real-captured videos.*

*Motivation: Recent advances in modeling 3D objects mostly rely on synthetic datasets due to the lack of large-scale real-scanned 3D databases. To facilitate the development of 3D perception, reconstruction, and generation in the real world, this paper proposes a large-scale object dataset.*

```latex
@article{wu2023omniobject3d,
  author = {Tong Wu, Jiarui Zhang, Xiao Fu, Yuxin Wang, Jiawei Ren, Liang Pan, Wayne Wu, Lei Yang, Jiaqi Wang, Chen Qian, Dahua Lin, Ziwei Liu},
  title = {OmniObject3D: Large-Vocabulary 3D Object Dataset for Realistic Perception, Reconstruction and Generation},
  journal={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```

**[CVPR2023]** MVImgNet: A Large-scale Dataset of Multi-view Images, [[Project](https://gaplab.cuhk.edu.cn/projects/MVImgNet/)], [[Paper](https://arxiv.org/pdf/2303.06042.pdf)]. 

*Keywords: release a large-scale dataset of multi-view images, 6.5 million frames from 219188 videos crossing objects from 238 classes; dervive a 3D object point cloud dataset, 150 categories, 87200 samples.*

*Motivation: Due to the laborious collection of real-world 3D data, there is no generic dataset serving as a counterpart of ImageNet in 3D vision, this paper introduces MVImgNet.* 

```latex
@inproceedings{yu2023mvimgnet,
    title     = {MVImgNet: A Large-scale Dataset of Multi-view Images},
    author    = {Yu, Xianggang and Xu, Mutian and Zhang, Yidan and Liu, Haolin and Ye, Chongjie and Wu, Yushuang and Yan, Zizheng and Liang, Tianyou and Chen, Guanying and Cui, Shuguang, and Han, Xiaoguang},
    booktitle = {CVPR},
    year      = {2023}
}
```
**[ICRA2022]** Google Scanned Objects: A High-Quality Dataset of 3D Scanned Household Items, [[Paper](https://arxiv.org/pdf/2204.11918.pdf)], [[Dataset](https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research)].

*Keywords: 3D scanned objects dataset; 1030 household objects.*

*Motivation: Handcrafted models built from polygons and primitives correspond poorly to real objects, and real-world data collection is challenging. This paper provides a large-scale 3D scanned objects dataset for public research.*

```latex
@INPROCEEDINGS{9811809,
  author={Downs, Laura and Francis, Anthony and Koenig, Nate and Kinman, Brandon and Hickman, Ryan and Reymann, Krista and McHugh, Thomas B. and Vanhoucke, Vincent},
  booktitle={2022 International Conference on Robotics and Automation (ICRA)}, 
  title={Google Scanned Objects: A High-Quality Dataset of 3D Scanned Household Items}, 
  year={2022},
  pages={2553-2560},
  doi={10.1109/ICRA46639.2022.9811809}
}
```

**[ECCV2022]** TO-Scene: A Large-scale Dataset for Understanding 3D Tabletop Scenes, [[Paper](https://arxiv.org/pdf/2203.09440.pdf)], [[Code](https://github.com/GAP-LAB-CUHK-SZ/TO-Scene)].

*Keywords: large-scale 3D dataset; table top scenes, contains 20740 scenes; objects(sim) from ModelNet and ShapeNet, 55 classes, 51300 models; tables(real-world) from ScanNet.*

```latex
@inproceedings{xu2022toscene,
  title={TO-Scene: A Large-scale Dataset for Understanding 3D Tabletop Scenes},
  author={Xu, Mutian and Chen, Pei and Liu, Haolin and Han, Xiaoguang},
  booktitle={ECCV},
  year={2022}
}
```

## 3. Grasp Detection
### 3.1 General grasping

**[ICRA2024]** ASGrasp: Generalizable Transparent Object Reconstruction and 6-DoF Grasp Detection from RGB-D Active Stereo Camera, [[Project](https://pku-epic.github.io/ASGrasp/)], [[Paper](https://arxiv.org/pdf/2405.05648)], [[Code](https://github.com/jun7-shi/ASGrasp)].

*Keywords: RGB+left/right IR input; handle diffuse, transparent and specular objects; 3D point cloud reconstruction first, then 6D grasp detection (GSNet); create an extensive synthetic dataset through domain randomization.*

**[RA-L2024]** RGBGrasp: Image-based Object Grasping by Capturing Multiple Views during Robot Arm Movement with Neural Radiance Fields, [[Paper](https://arxiv.org/pdf/2311.16592)].

*Keywords: 6-DoF grasp; use multi-view RGBs to reconstruct scene point cloud; integrate pre-trained monocular depth estimation network with NERF to achieve precise 3D reconstruction; use AnyGrasp to detect 6D grasps.*

```latex
@ARTICLE{10517376,
  author={Liu, Chang and Shi, Kejian and Zhou, Kaichen and Wang, Haoxiao and Zhang, Jiyao and Dong, Hao},
  journal={IEEE Robotics and Automation Letters},
  title={RGBGrasp: Image-based Object Grasping by Capturing Multiple Views during Robot Arm Movement with Neural Radiance Fields},
  year={2024},q
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/LRA.2024.3396101}
}
```

**[arXiv2024]** Rethinking 6-Dof Grasp Detection: A Flexible Framework for High-Quality Grasping, [[Paper](https://arxiv.org/pdf/2403.15054.pdf)].

*Keywords: 6-DoF grasp; propose a flexible framework capable of scene-level and target-oriented grasping.*

```latex
@article{tang2024rethinking,
  title={Rethinking 6-Dof Grasp Detection: A Flexible Framework for High-Quality Grasping},
  author={Tang, Wei and Chen, Siang and Xie, Pengwei and Hu, Dingchang and Yang, Wenming and Wang, Guijin},
  journal={arXiv preprint arXiv:2403.15054},
  year={2024}
}
```

**[RA-L2023]** Efficient Heatmap-Guided 6-Dof Grasp Detection in Cluttered Scenes, [[Paper](https://ieeexplore.ieee.org/document/10168242)], [[Code](https://github.com/THU-VCLab/HGGD)].

*Keywords: 6-DoF grasp; RGB-D input.*

```latex
@ARTICLE{10168242,
  author={Chen, Siang and Tang, Wei and Xie, Pengwei and Yang, Wenming and Wang, Guijin},
  journal={IEEE Robotics and Automation Letters},
  title={Efficient Heatmap-Guided 6-Dof Grasp Detection in Cluttered Scenes},
  year={2023},
  volume={8},
  number={8},
  pages={4895-4902},
  doi={10.1109/LRA.2023.3290513}
}

```

**[RA-L2023]** GPDAN: Grasp Pose Domain Adaptation Network for Sim-to-Real 6-DoF Object Grasping, [[Paper](https://ieeexplore.ieee.org/abstract/document/10153686)].

*Keywords: 6D grasp; sim-to-real domain adaptation, ACRONYM -> GraspNet-1Billion.*

```latex
@ARTICLE{10153686,
  author={Zheng, Liming and Ma, Wenxuan and Cai, Yinghao and Lu, Tao and Wang, Shuo},
  journal={IEEE Robotics and Automation Letters}, 
  title={GPDAN: Grasp Pose Domain Adaptation Network for Sim-to-Real 6-DoF Object Grasping}, 
  year={2023},
  volume={8},
  number={8},
  pages={4585-4592},
  doi={10.1109/LRA.2023.3286816}}
```

**[IROS2023]** Multi-Source Fusion for Voxel-Based 7-DoF Grasping Pose Estimation, [[Paper](https://ieeexplore.ieee.org/document/10341840)].

*Keywords: 6D grasp.*

```latex
@INPROCEEDINGS{10341840,
  author={Qiu, Junning and Wang, Fei and Dang, Zheng},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Multi-Source Fusion for Voxel-Based 7-DoF Grasping Pose Estimation}, 
  year={2023},
  pages={968-975},
  doi={10.1109/IROS55552.2023.10341840}}
```

**[arXiv2023]** Grasp-Anything: Large-scale Grasp Dataset from Foundation Models, [[Project](https://grasp-anything-2023.github.io/)], [[Paper](https://arxiv.org/pdf/2309.09818.pdf)], [[Code](https://github.com/andvg3/Grasp-Anything)].

*Keywords: 2D grasping; leverage knowledge from fundation models to generate a large-scale grasping dataset with 1M samples and 3M objects, substantially surpassing prior datasets in diversity and magnitude.*

```latex
@misc{vuong2023graspanything,
      title={Grasp-Anything: Large-scale Grasp Dataset from Foundation Models}, 
      author={An Dinh Vuong and Minh Nhat Vu and Hieu Le and Baoru Huang and Binh Huynh and Thieu Vo and Andreas Kugi and Anh Nguyen},
      year={2023},
      eprint={2309.09818},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

**[arXiv2023]** Learning Tri-mode Grasping for Ambidextrous Robot Picking, [[Paper](https://arxiv.org/pdf/2302.06431.pdf)].

*Keywords: ambidextrous robot picking; 6D grasp; parallel-jaw gripper grasp + suction grasp + push; cluttered scenes; cluttered scenes.*

*Motivation: the fusion of grasp and suction can expand the the range of objects that can be picked; the fusion of prehensile and nonprehensile action can expand the picking space of ambidextrous robot. Thus, this paper proposes Push-Grasp-Suction tri-mode grasping strategy.*

```latex
@misc{zhou2023learning,
      title={Learning Tri-mode Grasping for Ambidextrous Robot Picking}, 
      author={Chenlin Zhou and Peng Wang and Wei Wei and Guangyun Xu and Fuyu Li and Jia Sun},
      year={2023},
      eprint={2302.06431},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

**[ICRA2023]** GraspNeRF: Multiview-based 6-DoF Grasp Detection for Transparent and Specular Objects Using Generalizable NeRF, [[Project](https://pku-epic.github.io/GraspNeRF/)], [[Paper](https://arxiv.org/pdf/2210.06575.pdf)], [[Code](https://github.com/PKU-EPIC/GraspNeRF)].

*Keywords: 6D grasp; cluttered scene; for transparent and specular objects; input multi-view RGBs;  leverage generalizable neural radiance field to predict TSDF.*

*Motivation: To tackle 6-DoF grasp detection for transparent and specular objects, propose a multi-view RGB-based network, which can achieve material-agnostic object grasping in clutter.*

```latex
@article{Dai2023GraspNeRF,
  title={GraspNeRF: Multiview-based 6-DoF Grasp Detection for Transparent and Specular Objects Using Generalizable NeRF},
  author={Qiyu Dai and Yan Zhu and Yiran Geng and Ciyu Ruan and Jiazhao Zhang and He Wang},
  booktitle={IEEE International Confersence on Robotics and Automation (ICRA)},
  year={2023}
}
```

**[ICRA2023]** Keypoint-GraspNet: Keypoint-based 6-DoF Grasp Generation from the Monocular RGB-D input, [[Paper](https://arxiv.org/pdf/2209.08752.pdf)].

*Keywords: 6D grasp; RGB-D input; first detect grasp keypoints on RGB-D image, then recover the grasp poses with PnP algorithm; trained on synthetic dataset; faster than point cloud based methods.*

*Motivation: The point cloud based methods are prone to lead to failure on small objects, this paper explores 6-DoF grasp generation directly based on RGB-D image input.*

```latex
@article{chen2022keypoint,
  title={Keypoint-GraspNet: Keypoint-based 6-DoF Grasp Generation from the Monocular RGB-D input},
  author={Chen, Yiye and Lin, Yunzhi and Vela, Patricio},
  journal={arXiv preprint arXiv:2209.08752},
  year={2022}
}
```

**[ICRA2023]** RGB-D Grasp Detection via Depth Guided Learning with Cross-modal Attention, [[Paper](https://arxiv.org/pdf/2302.14264.pdf)].

*Keywords: 2D grasp; RGB-D input; Graspnet-1billion dataset; propose depth guided cross-modal attention network.*

*Motivation: The quality of depth maps captured by RGB-D sensors is relatively low, which makes obtaining grasping depth and multi-modal clues fusion challenging. To address the two issues, this paper proposes Depth Guided Cross-modal Attention Network.*

```latex
@article{qin2023rgb,
  title={RGB-D Grasp Detection via Depth Guided Learning with Cross-modal Attention},
  author={Qin, Ran and Ma, Haoxiang and Gao, Boyang and Huang, Di},
  journal={arXiv preprint arXiv:2302.14264},
  year={2023}
}
```

**[CoRL2022]** Towards Scale Balanced 6-DoF Grasp Detection in Cluttered Scenes, [[Paper](https://arxiv.org/pdf/2212.05275.pdf)], [[Code](https://github.com/mahaoxiang822/Scale-Balanced-Grasp)].

*Keywords: 6D grasp; cluttered scene; Graspnet-1billion dataset; data augmentation, mix the point cloud of syntetic data and real-scene data; pretrain an unseen point cloud instance segmentation network to generate masks for all objects, and then uniformly sample points from all objects for grasp learning; balance the grasp learning on different grasp width scale.*

*Motivation: To address the difficulty in dealing with small-scale samples.*

```latex
@InProceedings{Ma_2021_BMVC,
    author    = {Haoxiang, Ma and Huang, Di},
    title     = {Towards Scale Balanced 6-DoF Grasp Detection in Cluttered Scenes},
    booktitle = {Conference on Robot Learninsg (CoRL)},
    year      = {2022}
}
```

**[CoRL2022]** Volumetric-based Contact Point Detection for 7-DoF Grasping, [[Paper](https://openreview.net/pdf?id=SrSCqW4dq9)], [[Code](https://github.com/caijunhao/vcpd)].

*Keywords: 6D grasp; cluttered scenes; trained on synthetic data; TSDF-based; pipeline, multi-view fusion, contact-point sampling, evaluation and collision checking.*

```latex
@inproceedings{cai2022volumetric,
    title     = {Volumetric-based Contact Point Detection for 7-DoF Grasping},
    author    = {Cai, Junhao and Su, Jingcheng and Zhou, Zida and Cheng, Hui and Chen, Qifeng and Wang, Michael Yu},
    booktitle={Conference on Robot Learning (CoRL)},
    year={2022},
    organization={PMLR}
}
```

**[RA-L2022]** DA2 Dataset: Toward Dexterity-Aware Dual-Arm Grasping, [[Project](https://sites.google.com/view/da2dataset)], [[Paper](https://arxiv.org/pdf/2208.00408.pdf)], [[Code](https://github.com/ymxlzgy/DA2)].

*Keywords: 6D grasp; single object; dual-arm grasping for large objects in simulation; release a large scale dual-arm grasping dataset, 6327 objects, 9M grasp pairs.*

```latex
@article{da2dataset,
  author={Zhai, Guangyao and Zheng, Yu and Xu, Ziwei and Kong, Xin and Liu, Yong and Busam, Benjamin and Ren, Yi and Navab, Nassir and Zhang, Zhengyou},
  journal={IEEE Robotics and Automation Letters}, 
  title={DA$^2$ Dataset: Toward Dexterity-Aware Dual-Arm Grasping}, 
  year={2022},s
  volume={7},
  number={4},
  pages={8941-8948},
  doi={10.1109/LRA.2022.3189959}
}
```

**[RA-L2022]** End-to-End Learning to Grasp via Sampling From Object Point Clouds, [[Paper](https://arxiv.org/pdf/2203.05585.pdf)], [[Code](https://github.com/antoalli/L2G)].

*Keywords: 6D grasp; single object; point cloud input; combines a differentiable sampling strategy to identify the visible contact points, then use classifier and regressor to predict other contact point and grasp angle.*

```latex
@ARTICLE{9830843,
  author={Alliegro, Antonio and Rudorfer, Martin and Frattin, Fabio and Leonardis, Aleš and Tommasi, Tatiana},
  journal={IEEE Robotics and Automation Letters}, 
  title={End-to-End Learning to Grasp via Sampling From Object Point Clouds}, 
  year={2022},
  volume={7},
  number={4},
  pages={9865-9872},
  doi={10.1109/LRA.2022.3191183}
}
```

**[RA-L2022]** EfficientGrasp: A Unified Data-Efficient Learning to Grasp Method for Multi-Fingered Robot Hands, [[Paper](https://arxiv.org/pdf/2206.15159.pdf)].

*Keywords: single object grasping; multi-finger gripper; generalize to different types of robotic grippers;  uses fingertip workspace points set as the gripper attribute input, detect the contact points on object point cloud.*

```latex
@ARTICLE{9813387,
  author={Li, Kelin and Baron, Nicholas and Zhang, Xian and Rojas, Nicolas},
  journal={IEEE Robotics and Automation Letters}, 
  title={EfficientGrasp: A Unified Data-Efficient Learning to Grasp Method for Multi-Fingered Robot Hands}, 
  year={2022},
  volume={7},
  number={4},
  pages={8619-8626},
  doi={10.1109/LRA.2022.3187875}
}
```

**[RA-L2022]** SymmetryGrasp: Symmetry-Aware Antipodal Grasp Detection From Single-View RGB-D Images, [[Paper](https://ieeexplore.ieee.org/abstract/document/9919329)].

*Keywords: 6D grasp; input RGBD image; single view; Mask-RCNN for symmetric region detection on RGB-D image, tranform the RGBD region to point cloud and aplly PointNet++ for grasp detection.*

```latex
@ARTICLE{9919329,
  author={Shi, Yifei and Tang, Zixin and Cai, Xiangting and Zhang, Hongjia and Hu, Dewen and Xu, Xin},
  journal={IEEE Robotics and Automation Letters}, 
  title={SymmetryGrasp: Symmetry-Aware Antipodal Grasp Detection From Single-View RGB-D Images}, 
  year={2022},
  volume={7},
  number={4},
  pages={12235-12242},
  doi={10.1109/LRA.2022.3214785}
}
```

**[ECCV2022]** Domain Randomization-Enhanced Depth Simulation and Restoration for Perceiving and Grasping Specular and Transparent Objects, [[Paper](https://arxiv.org/pdf/2208.03792.pdf)], [[Code](https://github.com/PKU-EPIC/DREDS)].

*Keywords: Depth restoration for robotic grasping; Swin-Tiny backbone for depth restoration, two-stream net for rgb and depth feature extraction; graspnet-baseline for 6D grasp.*

```latex
@inproceedings{dai2022dreds,
    title={Domain Randomization-Enhanced Depth Simulation and Restoration for Perceiving and Grasping Specular and Transparent Objects},
    author={Dai, Qiyu and Zhang, Jiyao and Li, Qiwei and Wu, Tianhao and Dong, Hao and Liu, Ziyuan and Tan, Ping and Wang, He},
    booktitle={European Conference on Computer Vision (ECCV)},
    year={2022}
}
```

**[ECCV2022]** TransGrasp: Grasp Pose Estimation of a Category of Objects by Transferring Grasps from Only One Labeled Instance, [[Paper](https://arxiv.org/pdf/2207.07861.pdf)], [[Code](https://github.com/yanjh97/TransGrasp)].

*Keywords: 6D grasp; single object point cloud; from one instance to one category; 3 categories, objects model from ShapeNetCore; metric is grasp success rate in simulation environment; compare with GPD and 6-DOF GraspNet.*

```latex
@inproceedings{wen2022transgrasp,
  title={TransGrasp: Grasp Pose Estimation of a Category of Objects by Transferring Grasps from Only One Labeled Instance},
  author={Wen, Hongtao and Yan, Jianhang and Peng, Wanli and Sun, Yi},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

**[ICRA2022]** Hybrid Physical Metric For 6-DoF Grasp Pose Detection, [[Paper](https://arxiv.org/pdf/2206.11141.pdf)], [[Code](https://github.com/luyh20/FGC-GraspNet)].

*Keywords: 6D grasp; cluttered scene; real-world data; propose a new grasp score based on Graspnet-1billion, take force-closure metric, object flatness, gravity and collision into consideration.*

```latex
@INPROCEEDINGS{9811961,
  author={Lu, Yuhao and Deng, Beixing and Wang, Zhenyu and Zhi, Peiyuan and Li, Yali and Wang, Shengjin},
  booktitle={2022 International Conference on Robotics and Automation (ICRA)}, 
  title={Hybrid Physical Metric For 6-DoF Grasp Pose Detection}, 
  year={2022},
  pages={8238-8244},
  doi={10.1109/ICRA46639.2022.9811961}
}
```

**[ICRA2022]** Context-Aware Grasp Generation in Cluttered Scenes, [[Paper](https://ieeexplore.ieee.org/document/9811371/)].

*Keywords: 6D grasp; Graspnet-1billion dataset; cluttered scene; real-world data; pointnet++ backbone, vote and cluster seed points; self-attention mechanism for context learning.*

```latex
@INPROCEEDINGS{9811371,
  author={Hoang, Dinh-Cuong and Stork, Johannes A. and Stoyanov, Todor},
  booktitle={2022 International Conference on Robotics and Automation (ICRA)}, 
  title={Context-Aware Grasp Generation in Cluttered Scenes}, 
  year={2022},
  pages={1492-1498},
  doi={10.1109/ICRA46639.2022.9811371}
}
```

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
  doi={10.1109/LRA.2022.3142424}
}
```

**[arXiv2022]** A Robotic Visual Grasping Design: Rethinking Convolution Neural Network with High-Resolutions, [[Paper](https://arxiv.org/pdf/2209.07459.pdf)], [[Code](https://github.com/USTCzzl/HRG_Net)].

*Keywords: 2D planar grasp; Cornell and Jacquard grasping datasets; high-resolution CNN for feature extraction; D/RGB/RGB-D input.*

```latex
@article{zhou2022robotic,
  title={A Robotic Visual Grasping Design: Rethinking Convolution Neural Network with High-Resolutions},
  author={Zhou, Zhangli and Wang, Shaochen and Chen, Ziyang and Cai, Mingyu and Kan, Zhen},
  journal={arXiv preprint arXiv:2209.07459},
  year={2022}
}
```

**[RA-L2022]** When Transformer Meets Robotic Grasping: Exploits Context for Efficient Grasp Detection, [[Paper](https://arxiv.org/pdf/2202.11911.pdf)], [[Code](https://github.com/WangShaoSUN/grasp-transformer)].

*Keywords: 2D planar grasp; Cornell and Jacquard grasping datasets; cluttered scene; Transformer based architecture; D/RGB/RGB-D input.*

```latex
@ARTICLE{9810182,
  author={Wang, Shaochen and Zhou, Zhangli and Kan, Zhen},
  journal={IEEE Robotics and Automation Letters}, 
  title={When Transformer Meets Robotic Grasping: Exploits Context for Efficient Grasp Detection}, 
  year={2022},
  volume={7},
  number={3},
  pages={8170-8177},
  doi={10.1109/LRA.2022.3187261}
}
```

**[ICCV2021]** Graspness Discovery in Clutters for Fast and Accurate Grasp Detection,        [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Graspness_Discovery_in_Clutters_for_Fast_and_Accurate_Grasp_Detection_ICCV_2021_paper.pdf)],  [[Code(non-official)](https://github.com/rhett-chen/graspness_implementation)], [[Supp](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Wang_Graspness_Discovery_in_ICCV_2021_supplemental.zip)].

*Keywords: 6D general grasp; cluttered scene; real-world dataset GraspNet-1billion;  single-view scene point cloud input; MinkowskiEngine sparse convolution, ResUNet14.*

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

*Keywords: 6D grasp; cluttered scene; single-view scene point cloud input; real-world dataset GraspNet-1billion; jointly predict grasp poses, semantic segmentation and collision detection.*

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

*Keywords: 6D genral grasp; cluttered scene; RGB and single-view point cloud input; real-world dataset GraspNet-1billion.*

```latex
@INPROCEEDINGS{9561409,
  author={Gou, Minghao and Fang, Hao-Shu and Zhu, Zhanda and Xu, Sheng and Wang, Chenxi and Lu, Cewu},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={RGB Matters: Learning 7-DoF Grasp Poses on Monocular RGBD Images}, 
  year={2021},
  pages={13459-13466},
  doi={10.1109/ICRA48506.2021.9561409}
}
```

**[RA-L2021]** SuctionNet-1Billion: A Large-Scale Benchmark for Suction Grasping, [[Project](https://graspnet.net/suction)], [[Paper](https://arxiv.org/pdf/2103.12311.pdf)], [[Code](https://github.com/graspnet/suctionnet-baseline)].

*Keywords: suction; cluttered scene; RGBD input; release a large-scale real-world suction datatset.*

```latex
@ARTICLE{9547830,
  author={Cao, Hanwen and Fang, Hao-Shu and Liu, Wenhai and Lu, Cewu},
  journal={IEEE Robotics and Automation Letters}, 
  title={SuctionNet-1Billion: A Large-Scale Benchmark for Suction Grasping}, 
  year={2021},
  volume={6},
  number={4},
  pages={8718-8725},
  doi={10.1109/LRA.2021.3115406}
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
  pages={6350-6356},
  doi={10.1109/ICRA48506.2021.9562046}
}
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
  doi={10.1109/ICRA48506.2021.9561920}
}
```

**[ICRA2021]** Acronym: A large-scale grasp dataset based on simulation, [[Project](https://sites.google.com/nvidia.com/graspdataset)],   [[Paper](https://arxiv.org/pdf/2011.09584.pdf)], [[Code](https://github.com/NVlabs/acronym)].

*Keywords: 6D grasp; release a grasp dataset in simulation; 8872 objects, 262 categories, 17.7M grasps; in addition to single object, acronym also contains scenes with structured clutter.*

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
  pages={1532-1538},
  doi={10.1109/ICRA40945.2020.9197413}
}
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

**[CoRL2020]** S4G: Amodal Single-view Single-Shot SE(3) Grasp Detection in Cluttered Scenes, [[Project](https://sites.google.com/view/s4ggrapsing)], [[Paper](https://arxiv.org/pdf/1910.14218.pdf)], [[Code](https://github.com/yzqin/s4g-release)].

*Keywords: 6D grasp; cluttered scene; single-view point cloud input; one-stage grasp prediction; train on synthetic data.*

```latex
@inproceedings{qin2020s4g,
  title={S4g: Amodal Single-View Single-Shot SE(3) Grasp Detection in Cluttered Scenes},
  author={Qin, Yuzhe and Chen, Rui and Zhu, Hao and Song, Meng and Xu, Jing and Su, Hao},
  booktitle={Conference on Robot Learning},
  pages={53--65},
  year={2020},
  organization={PMLR}
}
```

**[IROS2020]** Antipodal Robotic Grasping using Generative Residual Convolutional Neural Network, [[Paper](https://arxiv.org/pdf/1909.04810.pdf)], [[Code](https://github.com/skumra/robotic-grasping)].

*Keywords: 2D grasp; cluttered scene; input RGB/D/RGB-D; Cornell dataset and Jacuard dataset.*

```latex
@INPROCEEDINGS{9340777,
  author={Kumra, Sulabh and Joshi, Shirin and Sahin, Ferat},
  booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Antipodal Robotic Grasping using Generative Residual Convolutional Neural Network}, 
  year={2020},
  pages={9626-9633},
  doi={10.1109/IROS45743.2020.9340777}
}
```

**[ICRA2020]** Using Synthetic Data and Deep Networks to Recognize Primitive Shapes for Object Grasping, [[Paper](https://arxiv.org/pdf/1909.08508.pdf)], [[Code](https://github.com/ivalab/grasp_primitiveShape)].

*Keywords: depth input; segment object into primitive shape classes, transform the predefined grasps on each primitive shape class to object.*

```latex
@INPROCEEDINGS{9197256,
  author={Lin, Yunzhi and Tang, Chao and Chu, Fu-Jen and Vela, Patricio A.},
  booktitle={2020 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={Using Synthetic Data and Deep Networks to Recognize Primitive Shapes for Object Grasping}, 
  year={2020},
  pages={10494-10501},
  doi={10.1109/ICRA40945.2020.9197256}
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
  doi={10.1109/ICRA40945.2020.9197415}
}
```

**[IJRR2020]** Learning robust, real-time, reactive robotic grasping, [[Paper](https://journals.sagepub.com/doi/full/10.1177/0278364919859066)], [[Code](https://github.com/dougsm/ggcnn)].

*Keywords: GG-CNN; depth image input.*

```latex
@article{doi:10.1177/0278364919859066,
  author = {Douglas Morrison and Peter Corke and Jürgen Leitner},
  title ={Learning robust, real-time, reactive robotic grasping},
  journal = {The International Journal of Robotics Research},
  volume = {39},
  number = {2-3},
  pages = {183-201},
  year = {2020},
  doi = {10.1177/0278364919859066},
}
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

**[RSS2018]** Closing the Loop for Robotic Grasping: A Real-time, Generative Grasp Synthesis Approach, [[Paper](https://arxiv.org/pdf/1804.05172.pdf)], [[Code](https://github.com/dougsm/ggcnn)].

```latex
@inproceedings{morrison2018closing,
	title={Closing the Loop for Robotic Grasping: A Real-time, Generative Grasp Synthesis Approach},
	author={Morrison, Douglas and Corke, Peter and Leitner, J\"urgen},
	booktitle={Proc.\ of Robotics: Science and Systems (RSS)},
	year={2018}
}
```

**[RSS2017]** Dex-Net 2.0: Deep Learning to Plan Robust Grasps with Synthetic Point Clouds and Analytic Grasp Metrics, [[Paper](https://arxiv.org/pdf/1703.09312.pdf)], [[Code](https://github.com/BerkeleyAutomation/dex-net)].

```latex
@inproceedings{mahler2017dex,
	title="Dex-Net 2.0: Deep Learning to Plan Robust Grasps with Synthetic Point Clouds and Analytic Grasp Metrics",
	author="Jeffrey {Mahler} and Jacky {Liang} and Sherdil {Niyaz} and Michael {Laskey} and Richard {Doan} and Xinyu {Liu} and Juan {Aparicio} and Ken {Goldberg}",
	booktitle="Robotics: Science and Systems (RSS)",
	volume="13",
	notes="Sourced from Microsoft Academic - https://academic.microsoft.com/paper/2600030077",
	year="2017"
}
```

**[IJRR2017]** Grasp Pose Detection in Point Clouds, [[Paper](https://arxiv.org/pdf/1706.09911.pdf)], [[Code](https://github.com/atenpas/gpd)].

*Keywords: 6D grasp; point cloud input; CNN based method.*

```latex
@article{ten2017grasp,
  title={Grasp pose detection in point clouds},
  author={ten Pas, Andreas and Gualtieri, Marcus and Saenko, Kate and Platt, Robert},
  journal={The International Journal of Robotics Research},
  volume={36},
  number={13-14},
  pages={1455--1473},
  year={2017},
  publisher={SAGE Publications Sage UK: London, England}
}
```

**[IJRR2015]** Deep Learning for Detecting Robotic Grasps, [[Paper](https://arxiv.org/pdf/1301.3592v6.pdf)].

*Keywords: 2D grasp; cluttered scene.*

```latex
@article{lenz2015deep,
  title={Deep learning for detecting robotic grasps},
  author={Lenz, Ian and Lee, Honglak and Saxena, Ashutosh},
  journal={The International Journal of Robotics Research},
  volume={34},
  number={4-5},
  pages={705--724},
  year={2015},
  publisher={SAGE Publications Sage UK: London, England}
}
```

**[ICRA2021]** Efficient grasping from RGBD images: Learning using a new rectangle representation, [[Paper](https://ieeexplore.ieee.org/document/5980145)].

```latex
@INPROCEEDINGS{5980145,
  author={Yun Jiang and Moseson, Stephen and Saxena, Ashutosh},
  booktitle={2011 IEEE International Conference on Robotics and Automation}, 
  title={Efficient grasping from RGBD images: Learning using a new rectangle representation}, 
  year={2011},
  pages={3304-3311},
  doi={10.1109/ICRA.2011.5980145}
}
```

### 3.2 Dexterous grasping

**[RA-L2024]** Grasp Multiple Objects with One Hand, [[Project](https://multigrasp.github.io/)], [[Paper](https://arxiv.org/pdf/2310.15599.pdf)], [[Code](https://github.com/MultiGrasp/MultiGrasp)]

*Keywords: dexterous grasping; grasp multiple objects; use diffusion-based method to generate a pre-grasp pose, and use RL-based method to learn reaching and lifting policy; propose a large-scale synthetic dataset comprising 90k diverse multi-object grasps, utilizing the Shadow Hand.*

```latex
@article{li2024grasp,
    author={Li, Yuyang and Liu, Bo and Geng, Yiran and Li, Puhao and Yang, Yaodong and Zhu, Yixin and Liu, Tengyu and
    Huang, Siyuan},
    title={Grasp Multiple Objects with One Hand},
    journal={IEEE Robotics and Automation Letters},
    volume={9},
    number={5},
    pages={4027-4034},
    year={2024},
    doi={10.1109/LRA.2024.3374190}
}
```

**[ICCV2023]** UniDexGrasp++: Improving Dexterous Grasping Policy Learning via Geometry-aware Curriculum and Iterative Generalist-Specialist Learning, [[Paper](https://arxiv.org/abs/2304.00464)], [[Code](https://github.com/PKU-EPIC/UniDexGrasp2)]

*Keywords: dexterous grasping; end-to-end grasp; SOTA*

```latex
@inproceedings{wan2023unidexgrasp++,
  title={Unidexgrasp++: Improving dexterous grasping policy learning via geometry-aware curriculum and iterative generalist-specialist learning},
  author={Wan, Weikang and Geng, Haoran and Liu, Yun and Shan, Zikang and Yang, Yaodong and Yi, Li and Wang, He},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3891--3902},
  year={2023}
}
```

**[CVPR2023]** UniDexGrasp: Universal Robotic Dexterous Grasping via Learning Diverse Proposal Generation and Goal-Conditioned Policy, [[Paper](https://arxiv.org/abs/2303.00938)], [[Code](https://github.com/PKU-EPIC/UniDexGrasp2)]

*Keywords: dexterous grasping; two-stage grasp;*

```latex
@inproceedings{xu2023unidexgrasp,
  title={Unidexgrasp: Universal robotic dexterous grasping via learning diverse proposal generation and goal-conditioned policy},
  author={Xu, Yinzhen and Wan, Weikang and Zhang, Jialiang and Liu, Haoran and Shan, Zikang and Shen, Hao and Wang, Ruicheng and Geng, Haoran and Weng, Yijia and Chen, Jiayi and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4737--4746},
  year={2023}
}
```

**[arXiv2023]** Generalized Anthropomorphic Functional Grasping with Minimal Demonstrations, [[Paper](http://export.arxiv.org/pdf/2303.17808v1)].

*Keywords: dexterous grasping; functional grasp; learn from human grasp demonstration for category-level objects; object reconstruction -> variational grasp sampler -> iterative grasp refinement; 10k synthesized functional grasp dataset.*

```latex
@misc{wei2023generalized,
    title={Generalized Anthropomorphic Functional Grasping with Minimal Demonstrations},
    author={Wei Wei and Peng Wang and Sizhe Wang},
    year={2023},
    eprint={2303.17808},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
}
```

**[ICRA2023]** DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects Based on Simulation, [[Project](https://pku-epic.github.io/DexGraspNet/)], [[Paper](https://arxiv.org/pdf/2210.02697.pdf)], [[Code](https://github.com/PKU-EPIC/DexGraspNet)], [[Dataset](https://mirrors.pku.edu.cn/dl-release/DexGraspNet-ICRA2023/)]. 

*Keywords: dexterous grasping, ShadowHand; release a large-scale dexterous grasping dataset in simulation, 5355 objects, 133 categories, 1.32M grasps.*

*Motivation: Dexterous grasping is much more under-explored than parallel grasping, partially due to the lack of a large-scale dataset. To accelerate the study of dexterous object manipulation, propose a large-scale grasping dataset.*

```latex
@article{wang2022dexgraspnet,
  title={DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects Based on Simulation},
  author={Wang, Ruicheng and Zhang, Jialiang and Chen, Jiayi and Xu, Yinzhen and Li, Puhao and Liu, Tengyu and Wang, He},
  journal={arXiv preprint arXiv:2210.02697},
  year={2022}
}
```

**[ICRA2023]** GenDexGrasp: Generalizable Dexterous Grasping, [[Project](https://tongclass.ac.cn/publication/2022/gendexgrasp/)], [[Paper](https://arxiv.org/pdf/2210.00722.pdf)], [[Code](https://github.com/tengyu-liu/GenDexGrasp)].

*Keywords: 6D grasp; single object; multi-hand grasp; first first generate hand-agnostic contact map for the given object, then optimize the hand pose to match the generated contact map; propose a synthetic large-scale multi-hand grasping dataset.*

*Motivation: Existing methods mostly focus on a specific type of robot hand, and oftentimes fail to rapidly generate diverse grasps with a high success rate. This paper leverages the contact map as a hand-agnostic intermediate representation and transfers among diverse multi-fingered robotic hands.*

```latex
@article{li2022gendexgrasp,
  title={GenDexGrasp: Generalizable Dexterous Grasping},
  author={Li, Puhao and Liu, Tengyu and Li, Yuyang and Zhu, Yixin and Yang, Yaodong and Huang, Siyuan},
  journal={arXiv preprint arXiv:2210.00722},
  year={2022}
}
```

**[ICRA2022]** HGC-Net: Deep Anthropomorphic Hand Grasping in Clutter, [[Paper](https://ieeexplore.ieee.org/document/9811756)], [[Code](https://github.com/yimingli1998/hgc_net)].

*Keywords: 6D grasp; cluttered scenes; dexterous grasping; single-view point cloud input; train on sythetic dataset.*

```latex
@INPROCEEDINGS{9811756,
  author={Li, Yiming and Wei, Wei and Li, Daheng and Wang, Peng and Li, Wanyi and Zhong, Jun},
  booktitle={2022 International Conference on Robotics and Automation (ICRA)}, 
  title={HGC-Net: Deep Anthropomorphic Hand Grasping in Clutter}, 
  year={2022},
  pages={714-720},
  doi={10.1109/ICRA46639.2022.9811756}
}
```

### 3.3 Semantic grasping 

**[IROS2023]** VL-Grasp: a 6-Dof Interactive Grasp Policy for Language-Oriented
Objects in Cluttered Indoor Scenes, [[Paper](https://arxiv.org/pdf/2308.00640)], [[Code](https://github.com/luyh20/VL-grasp)].

*Keywords: language-guided 6-DoF grasp; two-stage method, visual grounding first, then apply 6-DoF grasp pose detection; propose a Indoor Scenes visual grounding dataset.*

```latex
@INPROCEEDINGS{10341379,
  author={Lu, Yuhao and Fan, Yixuan and Deng, Beixing and Liu, Fangfu and Li, Yali and Wang, Shengjin},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={VL-Grasp: a 6-Dof Interactive Grasp Policy for Language-Oriented Objects in Cluttered Indoor Scenes}, 
  year={2023},
  pages={976-983},
  doi={10.1109/IROS55552.2023.10341379}
}

```

**[CoRL2023]** Language-guided Robot Grasping: CLIP-based Referring Grasp Synthesis in Clutter, [[Paper](https://openreview.net/pdf?id=j2AQ-WJ_ze)].

*Keywords: 2D grasp; RGB-Text input; develop a benchmark called OCID-VLG based on cluttered indoor scenes from OCID dataset; propose an end-to-end model to learn grasp synthesis directly from image-text pairs.*

```latex
@inproceedings{tziafas2023language,
  title={Language-guided Robot Grasping: CLIP-based Referring Grasp Synthesis in Clutter},
  author={Tziafas, Georgios and Yucheng, XU and Goel, Arushi and Kasaei, Mohammadreza and Li, Zhibin and Kasaei, Hamidreza},
  booktitle={7th Annual Conference on Robot Learning},
  year={2023}
}
```

**[ICRA2023]** A Joint Modeling of Vision-Language-Action for Target-oriented Grasping in Clutter, [[Paper](https://arxiv.org/pdf/2302.12610.pdf)].

*Keywords: language-guided task-oriented grasping; 6D grasp; cluttered scene; object-centric representation, a joint modeling of vision, language and grasp through cross-attention module; incorporate model-free reinforcement learning for obstacle removal and target object grasping; utilize priors from pre-trained CLIP and grasp model to improve the sample efficiency and alleviate the sim2real problem.*

```latex
@article{xu2023joint,
  title={A Joint Modeling of Vision-Language-Action for Target-oriented Grasping in Clutter},
  author={Xu, Kechun and Zhao, Shuqi and Zhou, Zhongxiang and Li, Zizhang and Pi, Huaijin and Zhu, Yifeng and Wang, Yue and Xiong, Rong},
  journal={arXiv preprint arXiv:2302.12610},
  year={2023}
}
```

**[arXiv2023]** Learning 6-DoF Fine-grained Grasp Detection Based on Part Affordance Grounding, [[Project](https://sites.google.com/view/lang-shape)], [[Paper](https://arxiv.org/pdf/2301.11564.pdf)].

*Keywords: single object; 6D grasp; fine-grained, task-oriented, language-guided grasp; propose a large language-guided shape grasping dataset, 16.6k objects of 16 categories in simulation environement; part affordance grounding and grasp stability evaluation; sampling-then-evaluation method.*

```latex
@article{song2023learning,
  title={Learning 6-DoF Fine-grained Grasp Detection Based on Part Affordance Grounding},
  author={Song, Yaoxian and Sun, Penglei and Ren, Yi and Zheng, Yu and Zhang, Yue},
  journal={arXiv preprint arXiv:2301.11564},
  year={2023}
}
```

**[IROS2023]** Task-Oriented Grasp Prediction with Visual-Language Inputs, [[Paper](https://arxiv.org/pdf/2302.14355.pdf)].

*Keywords: 2D grasp; cluttered scene; image and language input; two stage method, from object grouding to affordance grounding.*

```latex
@article{tang2023task,
  title={Task-Oriented Grasp Prediction with Visual-Language Inputs},
  author={Tang, Chao and Huang, Dehao and Meng, Lingxiao and Liu, Weiyu and Zhang, Hong},
  journal={arXiv preprint arXiv:2302.14355},
  year={2023}
}
```

**[IROS2022]** Learning 6-DoF Task-oriented Grasp Detection via Implicit Estimation and Visual Affordance, [[Paper](https://arxiv.org/pdf/2210.08537.pdf)].

*Keywords: task-oriented grasping; single object; point cloud input; 6D grasp; a grasping affordance detection module to generate grasps corresponding to affordance label, and a evaluation network to recognize success and faliure, a visual affordance network outputs affordance map to get fine grasp candidates.*

```latex
@INPROCEEDINGS{9981900,
  author={Chen, Wenkai and Liang, Hongzhuo and Chen, Zhaopeng and Sun, Fuchun and Zhang, Jianwei},
  booktitle={2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Learning 6-DoF Task-oriented Grasp Detection via Implicit Estimation and Visual Affordance}, 
  year={2022},
  pages={762-769},
  doi={10.1109/IROS47612.2022.9981900}
}
```

**[ICRA2023]** CoGrasp: 6-DoF Grasp Generation for Human-Robot Collaboration, [[Paper](https://arxiv.org/pdf/2210.03173.pdf)].

*Keywords: 6D grasp; RGB-D input, instance segmentation and get partial object point cloud, then shape completion, robot grasps and human grasps are generated based on completed object point cloud, finally a pruning network is applied to select the proper robot grasp compatible for the co-grasping.*

```latex
@INPROCEEDINGS{10160623,
  author={Keshari, Abhinav K. and Ren, Hanwen and Qureshi, Ahmed H.},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={CoGrasp: 6-DoF Grasp Generation for Human-Robot Collaboration}, 
  year={2023},
  pages={9829-9836},
  doi={10.1109/ICRA48891.2023.10160623}}
```

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
  doi={10.1109/LRA.2022.3142401}
}
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
  doi={10.1109/LRA.2022.3174648}
}
```

**[ICRA2022]** Learning Object Relations with Graph Neural Networks for Target-Driven Grasping in Dense Clutter, [[Project](https://sites.google.com/umn.edu/graph-grasping)], [[Paper](https://arxiv.org/pdf/2203.00875.pdf)].

*Keywords: target-driven grasp; cluttered scene; 6-D grasp; sampling based grasp generation; shape completion-assisted grasp sampling; formulate grasp graph, nodes representing object, edges indicating spatial relations between the objects; train on synthetic dataset; input scene RGB and query image.*

```latex
@INPROCEEDINGS{9811601,
  author={Lou, Xibai and Yang, Yang and Choi, Changhyun},
  booktitle={2022 International Conference on Robotics and Automation (ICRA)}, 
  title={Learning Object Relations with Graph Neural Networks for Target-Driven Grasping in Dense Clutter}, 
  year={2022},
  pages={742-748},
  doi={10.1109/ICRA46639.2022.9811601}
}
```

**[ICRA2022]** Interactive Robotic Grasping with Attribute-Guided Disambiguation,  [[Project](https://sites.google.com/umn.edu/attr-disam)], [[Paper](https://arxiv.org/pdf/2203.08037.pdf)].

*Keywords: cluttered scene; input scene RGBD and query language; 6D grasp; vision-and-language grounding module predicts target scores and attribute scores; attribute-guided  partially observable Markov decision process for language disambiguation(ask questions).*

```latex
@INPROCEEDINGS{9812360,
  author={Yang, Yang and Lou, Xibai and Choi, Changhyun},
  booktitle={2022 International Conference on Robotics and Automation (ICRA)}, 
  title={Interactive Robotic Grasping with Attribute-Guided Disambiguation}, 
  year={2022},
  pages={8914-8920},
  doi={10.1109/ICRA46639.2022.9812360}
}
```

**[ICRA2022]** I Know What You Draw: Learning Grasp Detection Conditioned on a Few Freehand Sketches, [[Project](https://hetolin.github.io/Skt_grasp/)], [[Paper](https://arxiv.org/pdf/2205.04026.pdf)].

*Keywords: 2D planar grasp; cluttered scene; target grasps by understanding freehand sketches; RGB image and graph-represented sketch input.*

```latex
@INPROCEEDINGS{9812372,
  author={Lin, Haitao and Cheang, Chilam and Fu, Yanwei and Xue, Xiangyang},
  booktitle={2022 International Conference on Robotics and Automation (ICRA)}, 
  title={I Know What You Draw: Learning Grasp Detection Conditioned on a Few Freehand Sketches}, 
  year={2022},
  pages={8417-8423},
  doi={10.1109/ICRA46639.2022.9812372}
}
```

**[ICRA2022]** Learning 6-DoF Object Poses to Grasp Category-level Objects by Language Instructions, [[Project](https://baboon527.github.io/lang_6d/)], [[Paper](https://arxiv.org/pdf/2205.04028v1.pdf)], [[Code](https://github.com/baboon527/lang_6d)].

*Keywords: grasp target object based on language description; two-stage method; 2D visual grounding, category-level object pose estimation; RGBD and language description input.*

```latex
@INPROCEEDINGS{9811367,
  author={Cheang, Chilam and Lin, Haitao and Fu, Yanwei and Xue, Xiangyang},
  booktitle={2022 International Conference on Robotics and Automation (ICRA)}, 
  title={Learning 6-DoF Object Poses to Grasp Category-Level Objects by Language Instructions}, 
  year={2022},
  pages={8476-8482},
  doi={10.1109/ICRA46639.2022.9811367}
}
```

**[ICRA2022]** CaTGrasp: Learning Category-Level Task-Relevant Grasping in Clutter from Simulation,  [[Project](https://sites.google.com/view/catgrasp)],  [[Paper](https://arxiv.org/pdf/2109.09163v1.pdf)],  [[Code](https://github.com/wenbowen123/catgrasp)].  

*Keywords: 6D category-level task-oriented grasp; cluttered scene in simulation;  self-supervised in simulation.*

```latex
@INPROCEEDINGS{9811568,
  author={Wen, Bowen and Lian, Wenzhao and Bekris, Kostas and Schaal, Stefan},
  booktitle={2022 International Conference on Robotics and Automation (ICRA)}, 
  title={CaTGrasp: Learning Category-Level Task-Relevant Grasping in Clutter from Simulation}, 
  year={2022},
  pages={6401-6408},
  doi={10.1109/ICRA46639.2022.9811568}
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

*Keywords: 2D grasp; joint grasp detection and semantic segmentation; OCID dataset.*

```latex
@INPROCEEDINGS{9561398,
  author={Ainetter, Stefan and Fraundorfer, Friedrich},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={End-to-end Trainable Deep Neural Network for Robotic Grasp Detection and Semantic Segmentation from RGB}, 
  year={2021},
  pages={13452-13458},
  doi={10.1109/ICRA48506.2021.9561398}
}
```

**[RSS2021]** INVIGORATE: Interactive Visual Grounding and Grasping in Clutter, [[Paper](http://roboticsproceedings.org/rss17/p020.pdf)].

*Keywords: input language expressions and RGB; cluttered scene; train separate neural networks for object detection, for visual grounding, for question generation, and for object blocking relationships detection and grasping.*

```latex
@INPROCEEDINGS{ZhangLu-RSS-21, 
    AUTHOR    = {Hanbo Zhang AND Yunfan Lu AND Cunjun Yu AND David Hsu AND Xuguang Lan AND Nanning Zheng}, 
    TITLE     = {INVIGORATE: Interactive Visual Grounding and Grasping in Clutter}, 
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
  doi={10.1109/ICRA40945.2020.9197318}
}
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

**[RSS2020]** Robot Object Retrieval with Contextual Natural Language Queries, [[Paper](https://arxiv.org/pdf/2006.13253.pdf)], [[Code](https://github.com/Thaonguyen3095/affordance-language)].

*Keywords: retrieval objects based on their usage; localize target object in RGB and then grasp it.*

```latex
@INPROCEEDINGS{Nguyen-RSS-20,
    AUTHOR    = {Thao Nguyen AND Nakul Gopalan AND Roma Patel AND Matthew Corsaro AND Ellie Pavlick AND Stefanie Tellex},
    TITLE     = {Robot Object Retrieval with Contextual Natural Language Queries},
    BOOKTITLE = {Proceedings of Robotics: Science and Systems},
    YEAR      = {2020},
    ADDRESS   = {Corvalis, Oregon, USA},
    MONTH     = {July},
    DOI       = {10.15607/RSS.2020.XVI.080}
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

**[ICRA2018]** Interactively Picking Real-World Objects with Unconstrained Spoken Language Instructions, [[Paper](https://arxiv.org/pdf/1710.06280.pdf)].

*Keywords: language-guided robotic grasping; resolve instruction ambiguity through dialogue; localize object by detection first, then identify the target object; vacuum gripper.*

```latex
@INPROCEEDINGS{8460699,
  author={Hatori, Jun and Kikuchi, Yuta and Kobayashi, Sosuke and Takahashi, Kuniyuki and Tsuboi, Yuta and Unno, Yuya and Ko, Wilson and Tan, Jethro},
  booktitle={2018 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={Interactively Picking Real-World Objects with Unconstrained Spoken Language Instructions}, 
  year={2018},
  pages={3774-3781},
  doi={10.1109/ICRA.2018.8460699}
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

### 3.4 Dynamic Grasping

**[IROS2023]** Flexible Handover with Real-Time Robust Dynamic Grasp Trajectory Generation, [[Paper](https://arxiv.org/pdf/2308.15622.pdf)].

*Keywords: flexible human-to-robot handover; generate object grasp trajectory based on grasp detection method GSNet and a lightweight transformer; future grasp prediction algorithm.*

```latex
@INPROCEEDINGS{10341777,
  author={Zhang, Gu and Fang, Hao-Shu and Fang, Hongjie and Lu, Cewu},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Flexible Handover with Real-Time Robust Dynamic Grasp Trajectory Generation}, 
  year={2023},
  pages={3192-3199},
  doi={10.1109/IROS55552.2023.10341777}}
```

**[CVPR2023]** Target referenced Reactive Grasping for Dynamic Objects, [[Project](https://graspnet.net/reactive)], [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Target-Referenced_Reactive_Grasping_for_Dynamic_Objects_CVPR_2023_paper.pdf)], [[Code](https://github.com/Todibo99/Target-referenced-Reactive-Grasping-for-Dynamic-Objects)].

*Keywords: reactive grasping, grasp dynalmic moving objects; 6D grasp; cluttered scenes; given grasps of first frame, tracking through generated grasp space; two-stage methods, first discover grasp correspndance between frames, then refine based on history information.*

*Motivation: current methods mainly focus on temporal smoothness but few consider their semantic consistency, can not guarantee the tracked grasps fall on the same part of same object. This paper propose a target-referenced setting to achieve temporally smooth and smeantically consistent reactive grasping in clutter given a targeted grasp.*

```latex
@InProceedings{Liu_2023_CVPR,
    author    = {Liu, Jirong and Zhang, Ruo and Fang, Hao-Shu and Gou, Minghao and Fang, Hongjie and Wang, Chenxi and Xu, Sheng and Yan, Hengxu and Lu, Cewu},
    title     = {Target-Referenced Reactive Grasping for Dynamic Objects},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {8824-8833}
}
```

**[T-RO2023]** AnyGrasp: Robust and Efficient Grasp Perception in Spatial and Temporal Domains, [[Project](https://graspnet.net/anygrasp.html)], [[Paper](https://arxiv.org/pdf/2212.08333.pdf)], [[Demo_Code](https://github.com/graspnet/anygrasp_sdk)].

*Keywords: 6D grasp; dynamic grasping;*

```latex
@ARTICLE{10167687,
  author={Fang, Hao-Shu and Wang, Chenxi and Fang, Hongjie and Gou, Minghao and Liu, Jirong and Yan, Hengxu and Liu, Wenhai and Xie, Yichen and Lu, Cewu},
  journal={IEEE Transactions on Robotics}, 
  title={AnyGrasp: Robust and Efficient Grasp Perception in Spatial and Temporal Domains}, 
  year={2023},
  volume={39},
  number={5},
  pages={3929-3945},
  doi={10.1109/TRO.2023.3281153}}
```

## 4. Research Groups  

- [SJTU Machine Vision and Intelligence Group](https://mvig.sjtu.edu.cn/),  Prof. Cewu Lu [[Google Scholar](https://scholar.google.de/citations?user=QZVQEWAAAAAJ&hl=zh-CN&oi=ao)].
- [UT Austin Robot Perception and Learning Lab](https://rpl.cs.utexas.edu/), Prof. Yuke Zhu [[Google Scholar](https://scholar.google.com/citations?user=mWGyYMsAAAAJ&hl=en)].
- [UCSD](https://xiaolonw.github.io/), Prof. Xiaolong Wang [[Google Scholar](https://scholar.google.com/citations?user=Y8O9N_0AAAAJ&hl=en)].
- [UCSD](https://cseweb.ucsd.edu//~haosu/), Prof. Hao Su [[Google Scholar](https://scholar.google.com/citations?user=1P8Zu04AAAAJ&hl=zh-CN)].
- [PKU-Agibot Lab](https://zsdonghao.github.io/), Prof. Hao Dong [[Google Scholar](https://scholar.google.com/citations?hl=en&user=xLFL4sMAAAAJ&view_op=list_works&sortby=pubdate)].
- [PKU EPIC Lab](https://hughw19.github.io/), Prof. He Wang [[Google Scholar](https://scholar.google.com/citations?user=roCAWkoAAAAJ&hl=en)].
- [ByteDance Research](https://www.taokong.org/), Prof. Tao Kong [[Google Scholar](https://scholar.google.com/citations?hl=en&user=kSUXLPkAAAAJ&view_op=list_works&sortby=pubdate)].
