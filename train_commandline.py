# Standard library imports
import sys

# own scripts
from wasabi.data import Data
from wasabi.trainer import Trainer
from wasabi.auxiliary_functions import load_config, save_config, load_config_from_model


def choose_path(argv_n):
    """

    :param argv_n: the single command line argument
    :return:
    """
    # load config file
    if isinstance(argv_n, str):
        if argv_n.split('.')[-1] == 'pt':
            config = load_config_from_model(argv_n)
            toggle_load = True
        elif argv_n.split('.')[-1] == 'yaml':
            config = load_config(argv_n)
            toggle_load = False
        else:
            raise ValueError('Could not recognize file ending of the config. It has to be .yaml for a config file '
                             'or .pt if you are loading from a model. '
                             'The filepath was {}'.format(argv_n))
    else:
        raise TypeError('The config keyword has to be either filled with a dictionary or a string.')
    return config, toggle_load


# set default parameters
config = None
n_epochs = 50
toggle_load = None
filepath = None
use_tb = True
alt_net = False


# get command line arguments
argv = sys.argv

# process input
for n in range(1, len(argv), 2):
    if argv[n] == '-f':
        filepath = argv[n+1]
        config, toggle_load = choose_path(argv[n+1])
    elif argv[n] == '-n':
        n_epochs = int(argv[n+1])
    elif argv[n] == '-tb':
        use_tb = bool(argv[n+1])
    elif argv[n] == '-l':
        alt_net = str(argv[n+1])
    elif argv[n] == '-h' or argv[n] == '--help':
        print("""help
                 -f <filepath>    The Filepath can be either to a config file or a existing model, in which case the
                                  model training will be continued.\n
                 -n <integer>     This is the number of epochs to trained.\n
                 -tb <True/False> Optional: Default True, will log additional parameters with TensorBoard.\n
                 -l <filepath>    Optional: This enables loading a trained net from the filepath while using the config from -f. \n""")
        sys.exit()
    else:
        raise('The argument {} is unknown. Use -h or --help to see a list of possible options'.format(argv[n]))

if toggle_load:
    data = Data()
    data.load_data_tensor(config, printing=False)
    trainer = Trainer(data, use_tb=use_tb)
    if alt_net:
        trainer.load_net(alt_net)
    else:
        trainer.load_net(filepath)
    trainer.train(n_epochs)
else:
    data = Data()
    data.load_data_tensor(config, printing=False)
    trainer = Trainer(data, use_tb=use_tb)
    if alt_net:
        trainer.load_net(alt_net)
    trainer.train(n_epochs)
