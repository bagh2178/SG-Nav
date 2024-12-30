# SG-Nav: Online 3D Scene Graph Prompting for LLM-based Zero-shot Object Navigation
### [Paper](https://arxiv.org/abs/2410.08189) | [Project Page](https://bagh2178.github.io/SG-Nav/) | [Video](https://cloud.tsinghua.edu.cn/f/ae050a060d624be4bc5d/?dl=1)

> SG-Nav: Online 3D Scene Graph Prompting for LLM-based Zero-shot Object Navigation  
> [Hang Yin](https://bagh2178.github.io/)*, [Xiuwei Xu](https://xuxw98.github.io/)\* $^\dagger$, [Zhenyu Wu](https://gary3410.github.io/), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), [Jiwen Lu](http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/)$^\ddagger$  

\* Equal contribution $\dagger$ Project leader $\ddagger$ Corresponding author


We propose a <b>zero-shot</b> object-goal navigation framework by constructing an online 3D scene graph to prompt LLMs. Our method can be directly applied to different kinds of scenes and categories <b>without training</b>. [中文解读](https://zhuanlan.zhihu.com/p/909651478).


## News
- [2024/12/30]: We update the code and simplify the installation.
- [2024/09/26]: SG-Nav is accepted to NeurIPS 2024!


## Demo
### Scene1:
![demo](./assets/demo1.gif)

### Scene2:
![demo](./assets/demo2.gif)

Demos are a little bit large; please wait a moment to load them. Welcome to the home page for more complete demos and detailed introductions.


## Method 

Method Pipeline:
![overview](./assets/pipeline.png)

## Installation

**Step 1 (Dataset)**

Download Matterport3D scene dataset from [here](https://niessner.github.io/Matterport/).
Download object-goal navigation episodes dataset from [here](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md).
According to your dataset path, set the scene dataset path (SCENES_DIR) and episode dataset path (DATA_PATH) in config file `configs/challenge_objectnav2021.local.rgbd.yaml`.

**Step 2 (Environment)**

Create conda environment with python==3.9.
```
conda create -n SG_Nav python==3.9
```

**Step 3 (Simulator)**

Install habitat-sim==0.2.4 and habitat-lab.
```
conda install habitat-sim==0.2.4 -c conda-forge -c aihabitat
pip install -e habitat-lab
```
Then replace the `agent/agent.py` in the installed habitat-sim package with `tools/agent.py` in our repository.
```
HABITAT_SIM_PATH=$(pip show habitat_sim | grep 'Location:' | awk '{print $2}')
cp tools/agent.py ${HABITAT_SIM_PATH}/habitat_sim/agent/
```

**Step 4 (Package)**

Install pytorch, pytorch3d and faiss. Install other packages.
```
conda install -c pytorch faiss-gpu=1.8.0
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

Install Grounded SAM.
```
pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
```

Install GLIP model and download GLIP checkpoint.
```
cd GLIP
python setup.py build develop --user
mkdir MODEL
cd MODEL
wget https://huggingface.co/GLIPModel/GLIP/resolve/main/glip_large_model.pth
cd ../../
```

Install Ollama.
```
curl -fsSL https://ollama.com/install.sh | sh
```

## Evaluation

Run SG-Nav:
```
python SG_Nav.py --visualize
```

## Citation
```
@article{yin2024sgnav, 
      title={SG-Nav: Online 3D Scene Graph Prompting for LLM-based Zero-shot Object Navigation}, 
      author={Hang Yin and Xiuwei Xu and Zhenyu Wu and Jie Zhou and Jiwen Lu},
      journal={arXiv preprint arXiv:2410.08189},
      year={2024}
}
```
