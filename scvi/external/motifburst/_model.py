import logging
import os
import pickle
from itertools import cycle
from typing import List, Optional

import numpy as np
import torch
from anndata import AnnData, read
from torch.utils.data import DataLoader

from scvi import _CONSTANTS, settings
from scvi.data import transfer_anndata_setup
from scvi.dataloaders import AnnDataLoader
from scvi.lightning import Trainer
from scvi.model._utils import _get_var_names_from_setup_anndata
from scvi.model.base import BaseModelClass, VAEMixin

from ._module import MotifBurstRNAModule
from ._task import MotifBurstRNATrainingPlan
