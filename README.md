# On the Geometry of Rectifier Convolutional Neural Networks
Describing the preimage of Conv+ReLU layers in terms of polytopes arrangements.

Recent work [1 ] describes the preimage of convolutional layers with ReLU activations in terms of polytopes in the preactivation space of the layer. When considering individual channels of convolutional kernels, the activation of each kernel can be described by mutual arrangements of polyhedral cones in the preimage space, with apices on the identity line.

The present code allows to train convolutional networks on CIFAR-10 and MNIST and then compute the discrete and continuous statistics that describe the polytope arrangements for each pair of stacked convolutional layers.

## Special convolutional layers

To restrict the study to the affine arrangement of polyhedral cones and make our theory as tight as possible, convolutional layers are implemented by swapping the order of cross-channel correlation and ReLU activation: 

![equation](https://latex.codecogs.com/svg.latex?%5Cbegin%7Bmultline%7D%20%5Clabel%7Beq%3Acustom-conv%7D%20%5Ctilde%7B%5Cmathcal%7BO%7D%7D%28o%2Ci%2Cj%29%20%3D%20%5C%5C%20b_o%20&plus;%20%5Csum%5Climits_%7Bc%20%3D%200%7D%5E%7Bn_%7B%5Ctext%7Bin%7D%7D%20-1%7D%20%5Cvarphi%20%5CBig%28%5Csum%5Climits_%7Bm%20%3D%200%7D%5E%7Bk%20-1%7D%20%5Csum%5Climits_%7Bn%20%3D%200%7D%5E%7Bk%20-1%7D%20%5Cmathcal%7BX%7D%28c%2C%20i&plus;m%2C%20j&plus;n%29%20%5Ccdot%20%5Cmathcal%7BW%7D_o%28c%2C%20m%2C%20n%29%5CBig%29%20%5C%5C%20%5Ctext%7Bfor%7D%20%5Cquad%20i%20%3D%200%2C%20%5Cldots%2C%20r%20-1%20%5Cquad%20%5Ctext%7Band%7D%20%5Cquad%20j%20%3D%200%2C%20%5Cldots%2C%20r%20-1%20%5Cend%7Bmultline%7D)

This allows to investigate for any bias arising from the optimization process in the arrangement of cones. All models that include the special convolutional layers are denoted ```Student_$arch``` where ```arch``` is one of the state of the art convolutional architectures. The list of supported models is defined in ```models.py``` and can be accessed by running ```train.py``` without the ```--arch``` option.

## Training models

To train our custom models, run ```train.py```, which supports two modes:
1. Training a model from scratch.
2. If a snapshot of a pretrained model is available, train a student model by knowledge distillation. This setting is useful to train the special models so that they mimic a standard convolutional teacher network.

```bash
usage: train.py [-h] [--arch ARCH] [--init INIT] [--kill-plateaus]
                [--runs RUNS] [--start-run START_RUN] [--dataset DATASET]
                [--dataset-path DATASET_PATH] [--noisy NOISY]
                [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--loss LOSS]
                [--optimizer OPTIMIZER] [--lr LR] [--lr-step LR_STEP]
                [--lr-decay LR_DECAY] [--weight-decay WEIGHT_DECAY]
                [--momentum MOMENTUM] [--path PATH]
                [--snapshot-every SNAPSHOT_EVERY] [--resume-from RESUME_FROM]
                [--temperature TEMPERATURE] [--alpha ALPHA] [--log LOG]

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH           Network architecture to be trained. Run without this
                        option to see a list of all supported archs.
  --init INIT           Weight initialization scheme [default = x_gaussian].
  --kill-plateaus       Quit training if the model plateaus in the first 10
                        epochs.
  --runs RUNS           Number of independent runs [default = 1]. Ignored when
                        using KD loss.
  --start-run START_RUN
                        Used to resume training, to skip already completed
                        runs.
  --dataset DATASET     Available datasets: mnist, cifar10.
  --dataset-path DATASET_PATH
                        Path where datasets are stored.
  --noisy NOISY         Percentage of corrupted labels [default = 0, max =
                        100].
  --epochs EPOCHS       The number of epochs used for training [default = 75].
  --batch-size BATCH_SIZE
                        The minibatch size for training [default = 100].
  --loss LOSS           Supported loss functions: softmax, KD].
  --optimizer OPTIMIZER
                        Supported optimizers: sgd, adam [default = sgd].
  --lr LR               The base learning rate for SGD optimization [default =
                        0.1].
  --lr-step LR_STEP     The step size (# iterations) of the learning rate
                        decay [default = 20].
  --lr-decay LR_DECAY   The decay factor of the learning rate decay [default =
                        0.1].
  --weight-decay WEIGHT_DECAY
                        The weight decay coefficient [default = 0.0005 ].
  --momentum MOMENTUM   The momentum coefficient for SGD [default = 0.9].
  --path PATH           The dirname where to store/load models [default =
                        './models'].
  --snapshot-every SNAPSHOT_EVERY
                        Snapshot the model state every E epochs [default = 0].
  --resume-from RESUME_FROM
                        Path to a model snapshot [default = None]. For KD
                        loss, this options specifies the teacher model file.
  --temperature TEMPERATURE
                        The softmax calibration factor [default = 22.0].
  --alpha ALPHA         Balance between KL and CrossEntropy terms in the KD
                        loss [default = 0.7].
  --log LOG             Logfile name [default = 'train.log'].

```

### Training from scratch

To train a model from scratch, run:

```bash
  python train.py --arch VGG11 \
                    --dataset "cifar10" \
                    --epochs 70 \
                    --loss "softmax" \
                    --weight-decay 5e-4 \
                    --lr 0.001 \
                    --log train.log \
                    --init x_uniform \
                    --snapshot-every 10 \
                    --runs 5
```

### Knowledge distillation

To train a network in the teacher/student setting, run:

```bash
  python train.py --arch Student_VGG11 --dataset cifar10 --epochs 70 --loss KD --resume-from models/cifar10/run_"$run"/"$arch"_"$epoch".tar --temperature 18 --alpha 0.7 --log student.log
```

## Polytope arrangements

We propose to describe inclusion and intersection of polytopes with 4 discrete states. Given two stacked convolutional layers L and L+1, for each convolutional filter ![equation](https://latex.codecogs.com/svg.latex?F_i%5E%7BL&plus;1%7D) of layer L+1, it is observed that:

* Each channel c of ![equation](https://latex.codecogs.com/svg.latex?F_i%5E%7BL&plus;1%7D) receives as input the c-th channel of the activation of layer L, which in turn is the result of the convolution of the c-th filter of layer L with its input.
* The mutual arrangement of the polytope ![equation](https://latex.codecogs.com/svg.latex?C_c%5E%7BL&plus;1%7D) corresponding to the c-th channel of ![equation](https://latex.codecogs.com/svg.latex?F_i%5E%7BL&plus;1%7D) with the polytopes corresponding to all channels of ![equation](https://latex.codecogs.com/svg.latex?F_c%5EL) is of our interest. This is computed by considering the conical (convex) combination of the polytopes corresponding to the channels of ![equation](https://latex.codecogs.com/svg.latex?F_c%5EL).

For the discrete case, 4 states are identified:
* OUT_FULL_IN: the cone ![equation](https://latex.codecogs.com/svg.latex?C_c%5E%7BL&plus;1%7D) is fully contained by the convex combination of cones of ![equation](https://latex.codecogs.com/svg.latex?F_c%5EL).
* OUT_PARTIAL_IN: the cone ![equation](https://latex.codecogs.com/svg.latex?C_c%5E%7BL&plus;1%7D) has partial non-empty intersection with the convex combination of cones of ![equation](https://latex.codecogs.com/svg.latex?F_c%5EL).
* IN_PARTIAL_OUT: the convex combination of cones of ![equation](https://latex.codecogs.com/svg.latex?F_c%5EL) is partially contained in ![equation](https://latex.codecogs.com/svg.latex?C_c%5E%7BL&plus;1%7D) (its vertex is inside the convex hull of ![equation](https://latex.codecogs.com/svg.latex?C_c%5E%7BL&plus;1%7D)).
* IN_FULL_OUT: the convex combination of cones of ![equation](https://latex.codecogs.com/svg.latex?F_c%5EL) is fully contained in ![equation](https://latex.codecogs.com/svg.latex?C_c%5E%7BL&plus;1%7D) (its vertex is inside the convex hull of ![equation](https://latex.codecogs.com/svg.latex?C_c%5E%7BL&plus;1%7D)).

Given a snapshot of a trained model, the continuous and discrete estimators of the polytope arrangements, for each pair of stacked convolutional layers, can be computed by running ```nesting.py```. The program can be run in parallel by using MPI.

The suggested number of parallel jobs N should divide the number of convolutional kernels N_L learned by each layer L.

```bash

nesting.py [-h] [--arch ARCH] [--dataset DATASET]
                  [--dataset-path DATASET_PATH] [--noisy NOISY] [--path PATH]
                  [--student STUDENT] [--log LOG] [--cuda] [--legacy]
                  [--quick]

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH           The teacher network architecture.
  --dataset DATASET     Available datasets: mnist, cifar10.
  --dataset-path DATASET_PATH
                        Path where datasets are stored.
  --noisy NOISY         Percentage of corrupted labels [default = 0, max =
                        100].
  --path PATH           The dirname where to store results [default =
                        './results'].
  --student STUDENT     Path to a student snapshot [default = None].
  --log LOG             Logfile name [default = 'nesting.log'].
  --cuda                Wheter to load the model on GPU.
  --legacy              Load legacy snapshots, stored with early versions of
                        train.py.
  --quick               Skip rotations, for faster runs.

```

Example:

```bash
  mpirun -n 16 python nesting.py --arch LeNet9 --dataset cifar10 --student LeNet9_snapshot.tar --cuda
```

The program generates a dictionary with the computed statistics, stored as a json file.

