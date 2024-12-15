
step 0: install package river, numpy, torch, torchvision, tqdm, torchvision by pip.

# result for table 1
step 1: run all the code in `cifar10_preprocess.ipynb`

step 2: run experiment by `python main.py --threshold -1 --continual True --dataset_name sine --classifier_name GNB`

The option for threhsold is -1, -3, -5

The option for classifier_name is GNB, DNN, hoeffding

The option for dataset_name is sine, sea0, sea10, sea20, mixed, powersupply, elec2, airline, cifar10

The option for continual is True, False

Note that for cifar10 experiment, the continual option must be True and the classifier_name must be resnet18

# result for table 2
step 1: run experimetn by `python main_ensemble.py --threshold -1 --dataset_name sine`

The option for threhsold is -1, -3, -5

The option for dataset_name is sine, sea0, sea10, sea20, mixed, powersupply, elec2, airline, cifar10

# result for illustrative exmaple

run `illustrative_example_fig1.ipynb`


All of the results will show in result.csv.
