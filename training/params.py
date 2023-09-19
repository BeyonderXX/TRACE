from model.Dynamic_network.PP import PP
from model.Dynamic_network.L2P import L2P
from model.Regular.LwF import LwF
from model.Regular.EWC import EWC
from model.Regular.GEM import GEM
from model.Regular.OGD import OGD
from model.Replay.MbPAplusplus import MbPAplusplus
from model.Replay.LFPT5 import LFPT5
from model.Regular.O_LoRA import O_LoRA
from model.base_model import CL_Base_Model
from model.lora import lora



Method2Class = {"PP":PP,
                "EWC":EWC,
                "GEM":GEM,
                "OGD":OGD,
                "LwF":LwF,
                "L2P":L2P,
                "MbPA++":MbPAplusplus,
                "LFPT5":LFPT5, 
                "O-LoRA":O_LoRA,
                "base":CL_Base_Model,
                "lora":lora}

AllDatasetName = ["C-STANCE","FOMC","MeetingBank","Papyrus-f","Py150","ScienceQA","ToolBench","NumGLUE-cm","NumGLUE-ds","20Minuten"]

