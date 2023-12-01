import torch
from dataclasses import dataclass, field
from typing import List, Tuple
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Database.database import database


ADNI = database('ADNI', 'ADNI')
PPMI = database('PPMI', 'PPMI')
ADNI_fMRI = database('ADNI_fMRI', 'ADNI_fMRI')
OCD_fMRI = database('OCD_fMRI', 'OCD_fMRI')
FTD_fMRI = database('FTD_fMRI', 'FTD_fMRI')


@dataclass
class AEConfig():
    name: str
    lr: float = 0.005
    weight_decay: float = 5e-4
    num_epochs: int = 1000
    seed: int = 24
    dropout: float = 0.05
    device: torch.device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size: int = 4
    slice_num: int = 100
    random_state: Tuple[int] = (0, 1, 2, 3, 4)
    add_terminal: bool = True
    num_layers: int = 3
    latent_dim: int = 16
    encoder_layers: List[int] = field(default_factory=lambda: [1024])
    decoder_layers: List[int] = field(init=False)
    loss_function: torch.nn = torch.nn.MSELoss()

    def __post_init__(self):
        self.decoder_layers = self.encoder_layers[::-1]


ADNI_config = AEConfig('ADNI')
PPMI_config = AEConfig('PPMI')
ADNI_fMRI_config = AEConfig('ADNI_fMRI')
OCD_fMRI_config = AEConfig('OCD_fMRI')
FTD_fMRI_config = AEConfig('FTD_fMRI')
