The instruction for implementation of SG-Nav for object-goal navigation on MP3D dataset. 
## Installation

**Step 1**
Download Matterport3D scene dataset from [here](https://niessner.github.io/Matterport/).
Download object-goal navigation episodes dataset from [here](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md).
According to your dataset path, set the scene dataset path (SCENES_DIR) and episode dataset path (DATA_PATH) in config file `configs/challenge_objectnav2021.local.rgbd.yaml`.

**Step 2**
Install habitat-sim==0.2.4 with `conda install habitat-sim==0.2.4 -c conda-forge -c aihabitat`.
Replace the `agent/agent.py` in the installed habitat-sim package with `tools/agent.py` in our repository.
Install habitat-lab with ``pip install -e habitat-lab``.

**Step 3**
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

**Step 4**
Install Grounded SAM according to [here](https://github.com/IDEA-Research/Grounded-Segment-Anything).

In `scenegraph.py`, change the path `'/path/to/Grounded-Segment-Anything/'` to your installation path of Grounded SAM.

**Step 5**
Install pytorch3d and faiss.
```
conda install -c pytorch faiss-gpu=1.8.0
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```
Install the packages.
```
pip install -r requirements.txt
```

**Step 6**
Install Ollama with `curl -fsSL https://ollama.com/install.sh | sh`. Deploy llama3.2 as the LLM and VLM.
