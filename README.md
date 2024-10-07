The instruction for implementation of SG-Nav for object-goal navigation on MP3D dataset. 

### Step 1
Download Matterport3D scene dataset from this [link](https://niessner.github.io/Matterport/).
Download object-goal navigation episodes dataset from this [link](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md)
Then change the scene dataset path(SCENES_DIR) and episode dataset path(DATA_PATH) in config file `configs/challenge_objectnav2021.local.rgbd.yaml`.

### Step 2
Install [habitat-sim](https://github.com/facebookresearch/habitat-sim) and [habitat-lab](https://github.com/facebookresearch/habitat-lab) according to their github page.

### Step 3
Install conda environment
```
conda env create -f SG_Nav.yml
```

### Step 4
Install GLIP
```
cd GLIP
python setup.py build develop --user
cd MODEL
wget https://huggingface.co/GLIPModel/GLIP/resolve/main/glip_large_model.pth
```

### Step 5
Install [ConceptGraph](https://github.com/concept-graphs/concept-graphs) according to the github.

### Step 6
Run SG-Nav on MP3D
```
python SG_Nav.py --evaluation local --reasoning both
```
Or run SG-Nav with multiprocess

```
python start_multiprocess.py
```

