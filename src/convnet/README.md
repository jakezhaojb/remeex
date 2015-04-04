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

### Start to train Convnet
    cd remeex/src/convnet
    th step0.lua --devid 1 --dataset osecond --nPlanes 128-256-256-512 --kSizes 255-127-127-95 --poolSizes 43-8-8-2 --stride 1-1-1-1 --lr 0.001 --mom 0.5 --batchsize 32


Notices and TODO
---
> Any feature can be adapted to this framework.
