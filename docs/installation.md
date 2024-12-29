The instruction for implementation of SG-Nav for object-goal navigation on MP3D dataset. 
## Installation

**Step 1**
Download Matterport3D scene dataset from [here](https://niessner.github.io/Matterport/).
Download object-goal navigation episodes dataset from [here](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md).
According to your dataset path, set the scene dataset path (SCENES_DIR) and episode dataset path (DATA_PATH) in config file `configs/challenge_objectnav2021.local.rgbd.yaml`.

**Step 2**
Install habitat-sim==0.2.4 according to [here](https://github.com/facebookresearch/habitat-sim).
Install habitat-lab with ``pip install -e habitat-lab``.

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
Install Grounded SAM according to [here](https://github.com/IDEA-Research/Grounded-Segment-Anything).

In `scenegraph.py`, change the path `'/path/to/Grounded-Segment-Anything/'` to your installation path of Grounded SAM.

**Step 6**
Install Ollama from [here](https://ollama.com/). Deploy llama3.3 70b as the LLM and llama3.2-vision 90b as the VLM. Replace the `LLM_Client` and `VLM_Client` in `scenegraph.py` with Ollama.
