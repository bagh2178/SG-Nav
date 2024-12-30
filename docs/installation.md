The instruction for implementation of SG-Nav for object-goal navigation on MP3D dataset. 
## Installation

**Step 1**
Download Matterport3D scene dataset from [here](https://niessner.github.io/Matterport/).
Download object-goal navigation episodes dataset from [here](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md).
According to your dataset path, set the scene dataset path (SCENES_DIR) and episode dataset path (DATA_PATH) in config file `configs/challenge_objectnav2021.local.rgbd.yaml`.

**Step 2**
Create conda environment with python==3.9.
```
conda create -n SG_Nav python==3.9
```

**Step 3**
Install habitat-sim==0.2.4 and habitat-lab.
```
conda install habitat-sim==0.2.4 -c conda-forge -c aihabitat
pip install -e habitat-lab
```
Then replace the `agent/agent.py` in the installed habitat-sim package with `tools/agent.py` in our repository.
```
cp tools/agent.py /path/to/your/conda/environment/SG_Nav/lib/python3.9/site-packages/habitat_sim-0.2.4-py3.9-linux-x86_64.egg/habitat_sim/agent/
```

**Step 4**
Install Grounded SAM.

```
pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
```

**Step 5**
Install pytorch3d and faiss, install packages. (Note: Do not change the version specified in `requirements.txt`)
```
conda install -c pytorch faiss-gpu=1.8.0
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install -r requirements.txt
```

**Step 6**
Install GLIP model and download GLIP checkpoint.
```
cd GLIP
python setup.py build develop --user
mkdir MODEL
cd MODEL
wget https://huggingface.co/GLIPModel/GLIP/resolve/main/glip_large_model.pth
cd ../../
```

**Step 7**
Install Ollama.
```
curl -fsSL https://ollama.com/install.sh | sh
```
