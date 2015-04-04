Convent on melody
===

Introduction
---
Coming soon.

Steps of training Convnet
---
### Feature extraction
    cd remeex/src/data
    python main.py

### Convert .csv to .t7b
    cd remeex/src/aux
    python main.py /path/to/data/folder /path/to/target/folder /path/to/t7b/folder

### Training Look-up table generated
    cd remeex/src/aux
    python table_generator.py /path/to/folder /path/to/save /name/of/table

### Start to traing Convnet
    cd remeex/src/convnet
    th step0.lua --devid 2 --dataset osecond --nPlanes 512-512-512 --kSize 256-256-256 --poolSize 43-16-8 --lr 0.001 --mon 0.5


Notices and TODO
---
1. Bugs are not all cleared.
2. Large scale experiments are not tested.
3. Any feature can be adapted to this framework.
4. Plan to test it on GEFORCE GTX 580, where cudnn is not supported because of arch. In this case, Temporal functions are faster.
