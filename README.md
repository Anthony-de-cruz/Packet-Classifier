# Encrypted Packet Classifier

Scripts to train and evaluate a Resnet-50 based neural network to classify encrypted network packets. 

## Setup & Execution

### Via [uv](https://docs.astral.sh/uv/) (Recommended)

```sh
uv sync
uv run train.py && uv run test.py
```

### Native

Unix:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py && python test.py
```

Windows:

```ps1
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train.py && python test.py
```
