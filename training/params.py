from model.Dynamic_network.PP import PP, convert_model
from model.Regular.LwF import LwF
from model.Regular.EWC import EWC
from model.Regular.GEM import GEM
from model.Regular.OGD import OGD


Method2Class = {"PP":PP,
                "EWC":EWC,
                "GEM":GEM,
                "OGD":OGD,
                "LwF":LwF}

AllDatasetName = ["C-STANCE","FOMC","MeetingBank","Papyrus-f","Py150","ScienceQA","ToolBench","NumGLUE-cm","NumGLUE-ds"]

