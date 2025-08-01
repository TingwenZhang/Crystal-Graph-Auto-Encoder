# Crystal Graph Auto-Encoder
A foundational model to form compact representation of crystals.

## File Structure
```
├── M_FEATURE_TABLE.pt
├── README.md
├── cif-files
│   ├── test
│   └── train
├── dataset.py
├── edge_bce.png
├── edge_feat.png
├── environment.yml
├── gae.pt
├── gae.py
├── main.ipynb
├── node_loss.png
├── split.py
└── total.png
```
`M_FEATURE_TABLE.pt`: chemical features of all elements

`cif-files`: training and validation Crystallograph Information Files

`dataset.py`: a class to store the `torch_geometric.data.Data` representation of `ase.Atoms`

`gae.pt`: trained auto-encoder

`main.ipynb`: running the auto-encoder

`split.py`: split `cif-files` into training and test set

## Create a virtual environment
`conda create -f environment.yml`

## Run the auto-encoder
Go to `main.ipynb`. The paper is at https://www.overleaf.com/read/ygcvtsfwtqbs#91eb2c. This project is still under development.