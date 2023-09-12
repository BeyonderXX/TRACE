from model.Dynamic_network.PP import PP, convert_model
from model.Regular.LWF import LwF
from model.Regular.EWC import EWC
from model.Regular.GEM import GEM
from model.Regular.OGD import OGD
from model.Replay.MbPAplusplus import MbPAplusplus
from model.Replay.LFPT5 import LFPT5
from model.Regular.O_LoRA import O_LoRA


Method2Class = {"PP":PP,
                "EWC":EWC,
                "GEM":GEM,
                "OGD":OGD,
                "LwF":LwF,
                "MbPA++":MbPAplusplus,
                "LFPT5":LFPT5, 
                "O-LoRA":O_LoRA}

AllDatasetName = ["C-STANCE","FOMC","MeetingBank","Papyrus-f","Py150","ScienceQA","ToolBench","NumGLUE-cm","NumGLUE-ds"]

