##
Installation
```bash
pip install osrl-lib
```

## Offline Policy Learning 
Depending on the Lagrangian mulplier you want to use, please specify the value for --multiplier.

```bash
python train/train_sosrl.py --task OfflineCarCircle-v0 --multiplier 0   #\lambda = 0
```

```bash
python train/train_sosrl.py --task OfflineCarCircle-v0 --multiplier 0.5   #\lambda = 0.5
```

## Online Policy Evaluation
```bash
python eval/eval_bcql.py --path path_to_model --eval_episodes 20
```