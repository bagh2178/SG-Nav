The instruction for implementation of SG-Nav for object-goal navigation on MP3D dataset. 
## Installation

**Step 1**
Download Matterport3D scene dataset from [here](https://niessner.github.io/Matterport/).
Download object-goal navigation episodes dataset from [here](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md).
According to your dataset path, set the scene dataset path (SCENES_DIR) and episode dataset path (DATA_PATH) in config file `configs/challenge_objectnav2021.local.rgbd.yaml`.

**Step 2**
Install habitat-sim according to [here](https://github.com/facebookresearch/habitat-sim).
Install habitat-lab according to [here](https://github.com/facebookresearch/habitat-lab) or install habitat-lab with ``pip install -e habitat-lab``.

**Step 3**
Install the conda environment we provided.
```
conda env create -f SG_Nav.yml
```

**Step 4**
Install GLIP model.
```
cd GLIP
python setup.py build develop --user
```
Download GLIP checkpoint.
```
cd MODEL
wget https://huggingface.co/GLIPModel/GLIP/resolve/main/glip_large_model.pth
```

**Step 5**
Install ConceptGraph according to [here](https://github.com/concept-graphs/concept-graphs).

