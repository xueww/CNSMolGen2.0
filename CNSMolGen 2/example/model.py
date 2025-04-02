# adds the paths to the repo folder
import sys
sys.path.insert(1, '../preprocessing/') # preprocessing code
sys.path.insert(1, '../experiments/') # folder with configuration files
sys.path.insert(1, '../data/') # data code
sys.path.insert(1, '../model/') # model code
sys.path.insert(1, '../evaluation/') # model code
import os
# repo modules
from main_preprocessor import preprocess_data #for data preprocessing
import configparser # to automatically change the .ini file
from sample import Sampler

# other modules
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw # for molecule depiction
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Scaffolds import MurckoScaffold
import ChemTools
# %load_ext autoreload
# %autoreload 2
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from trainer import Trainer

t = Trainer(experiment_name = 'BIMODAL_random')
t.cross_validation(stor_dir = '../evaluation/', restart = False)