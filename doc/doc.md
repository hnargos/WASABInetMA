# Documentation

The documentation for WASABInet.  
The idea behind WASABInet is to enable reproducible training of neural nets
which analyze z-spectra and return dB0, B1, T1 and T2 as well as their uncertainties.


## The 3 Classes

The python files data.py, trainer.py and eval_wasabi.py each contain a class 
which together provide the main functionality of this repo. Each function contains
a doc string for more details than presented here.

### Configuration Files

At the base of every call to the three classes stands a config file. An explanation 
what is possible with a config file can be found in an explanatory config file
`WASABInet/doc/net_config_explanation.yaml.`

### Data()

This is likely the first class one will access. It contains and preprocesses 
all the training, validation and test data. Currently, it can load BMCTool 
[n_Samples, 31_offsets] data, scanner data with the same dimensions as the BMCTool data
and BMCPhantom data.

```python
# import
from wasabi.data import Data 
from wasabi.auxiliary_functions import load_config

# load config
config = load_config(str("path_to_a_config"))

# load BMCTool data
data = Data()
data.load_data_tensor(config=config)

## or load BMCPhantom data
data2 = Data()
data2.load_phantom(filepath=str("path_to_a_phantom"),config=config)

## or load scanner data without target values
data3 = Data()
data3.load_data_tensor(config=config,     
                       x_file=str("scanner_data"),
                       evaluation=True,
                       add_noise=False,
                       create_zeros_tgts=True)

## or load scanner data with target values, for example from analytical solutions
# those targets have to follow the parameter order in the config file
data4 = Data()
data4.load_data_tensor(config=config,
                       x_file=str("scanner_data"),
                       y_file=str("analytical_targets"),
                       evaluation=True,
                       add_noise=False)
```

### Trainer()

The Trainer class performs the training and contains the functions to save 
and load the neural net.

```python
# import
from wasabi.trainer import Trainer
from wasabi.data import Data 
from wasabi.auxiliary_functions import load_config

# load config
config = load_config(str("path_to_a_config"))

# load any kind of data object from above
data = Data()
data.load_data_tensor(config=config)

# initialize trainer
trainer = Trainer(data=data)

## or initialize trainer with tensorboard statistics
trainer2 = Trainer(data=data, use_tb=True)

# perform training
trainer.train(n_epochs=5)

# save net beyond the autosave
trainer.save_net(str("filepath.pt"))
```

To load a pre-trained network one has to go through the same steps.
```python
from wasabi.trainer import Trainer
from wasabi.data import Data 
from wasabi.auxiliary_functions import load_config_from_model

# the filepath to a pre-trained model
filepath = str("filepath.pt")

# load config
config = load_config_from_model(filepath)

# load any kind of data object from above
data = Data()
data.load_data_tensor(config=config)

# initialize trainer
trainer = Trainer(data=data)

# load pre-trained network
trainer.load_net(filepath)
```

### EvalWasabi

This class provides the tools and interfaces to evaluate and predict a trained
network.

```python
from wasabi.eval_wasabi import EvalWasabi

# the filepath to a pre-trained model
filepath = str("filepath.pt")
# or a config file
filepath2 = str("filepath.yaml")

# initialize the evaluation object from a pre-trained model
# this loads the first "*best_model_save.pt" in the same folder as 
#  the filepath. This can differ from the .pt file given in "trainer="
eval_wasabi = EvalWasabi(trainer=str(filepath), data_type=str("BMCTool"))

# to specify a network to load use
eval_wasabi2 = EvalWasabi(trainer=str(filepath), 
                         data_type=str("BMCTool"),
                         net_to_load=str("other_filepath.pt"))


## or initialize the evaluation object with a pre-exiting trainer
from wasabi.trainer import Trainer
from wasabi.data import Data 
from wasabi.auxiliary_functions import load_config_from_model

# load config
config = load_config_from_model(filepath)

# load any kind of data object from above
data = Data()
data.load_data_tensor(config=config)

# initialize trainer
trainer = Trainer(data=data)

# initialize the EvalWasabi class with the trainer object
eval_wasabi3 = EvalWasabi(trainer=trainer, data_type=str("BMCTool"))
```
After initialization the first thing to be done is to predict the data.
```python
from wasabi.eval_wasabi import EvalWasabi

eval_wasabi = EvalWasabi(trainer=str("filepath.pt"), data_type=str("BMCTool"))
eval_wasabi.predict()
```
Now one can access the basic functions to visualize the predictions, the quality 
of predictions and get the parameter maps as a numpy array.

## Tensorboard

When "use_tb" is set to true, the training logs a few more parameters during training 
into a sub-folder, called tensorboard, of the output path via tensorboard. In auxiliary_functions.py 
is a function zip_and_del_tb(filepath) which checks the filepath for the tensorboard 
folder and zips it up and deletes the unzipped folder.