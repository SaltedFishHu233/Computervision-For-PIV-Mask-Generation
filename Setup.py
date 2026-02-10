import os
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast

repo_id="ShilongLiu/GroundingDINO"
filename="groundingdino_swinb_cogcoor.pth"
config_filename="GroundingDINO_SwinB.cfg.py"
device='cpu'
Directory=os.path.join(os.getcwd(),"Model")
GDD=os.path.join(Directory,"GD")
BBUD=os.path.join(Directory,"B")
hf_hub_download(repo_id=repo_id, filename=config_filename,local_dir=GDD)
hf_hub_download(repo_id=repo_id, filename=filename,local_dir=GDD)
BertModel.from_pretrained("bert-base-uncased",cache_dir=BBUD)
AutoTokenizer.from_pretrained("bert-base-uncased",cache_dir=BBUD)


print(GDD)
