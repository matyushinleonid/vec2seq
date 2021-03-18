# Vec2Seq: Symbolic Regression Model

:feelsgood:

# How To Run

## Data structure

```bash
.
└── vec2seq/
    ├── data/
    │   ├── polynomes/
    │   │   ├── train_formulae.txt
    │   │   ├── val_formulae.txt
    │   │   ├── val_formulae.txt
    │   │   ├── train_gp.txt
    │   │   ├── val_gp.txt
    │   │   └── test_gp.txt
    │   └── ...
    ├── ...
    └── ...
```


`*_formulae.txt` have the following structure (formulae, "polish" notation):
```bash
0,add INT+ 1 add mul INT- 3 pow x INT+ 2 add mul INT- 2 x mul INT+ 3 pow x INT+ 4
1,add INT+ 2 add mul INT- 4 pow x INT+ 2 add mul INT- 4 pow x INT+ 3 add mul INT- 4 pow x INT+ 4 mul INT+ 4 x
2,add INT+ 4 add pow x INT+ 3 add pow x INT+ 4 add mul INT- 1 pow x INT+ 2 mul INT- 5 x
...
```

`*_gp.txt` have the following structure:
```bash
0,14.678,-56.98,...,132.999
1,-0.17,5.41,...,-2.44
2,11.58,-111.38,...,2.999
...
```

## Running the scripts

Training the model:
```bash
python train.py --config configs/polynomes.yaml
```

Benching the model:

```bash
python test.py --checkpoint_path logs/polynomes/default/version_0/checkpoints/epoch=0-step=0.ckpt
```

After this you may collect an output at `logs/polynomes/default/version_0/epoch=0-step=0_output.csv` 