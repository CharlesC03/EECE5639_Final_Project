# convert_checkpoint.py
import torch
import hydra
from omegaconf import OmegaConf
from plonk.models.pretrained_models import Plonk

CHECKPOINT_PATH = "plonk/checkpoints/StreetCLIP_Att/last.ckpt"
EXP = "osv_5m_streetclip_multi_mean_r100"

SAVE_NAME = "StreetCLIP_Multi_Mean_Model"

# Load your hydra config to get the network architecture
hydra.initialize(version_base=None, config_path="../plonk/configs")

cfg = hydra.compose(
    config_name="config",
    overrides=[f"exp={EXP}"],  # your experiment config
)

network_config = cfg.model.network
serialized = OmegaConf.to_container(network_config, resolve=True)
del serialized["_target_"]

# Build the model
model = Plonk(**serialized)

# Load weights from Lightning checkpoint (extracts ema_network weights)
ckpt = torch.load(f"{CHECKPOINT_PATH}", map_location="cpu", weights_only=False)
state_dict = ckpt["state_dict"]
state_dict = {k: v for k, v in state_dict.items() if "ema_network" in k}
state_dict = {k.replace("ema_network.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

# Save in HuggingFace format (config saves architecture params for from_pretrained)
model.save_pretrained(f"local_models/{SAVE_NAME}", config=serialized)
print(f"Saved to local_models/{SAVE_NAME}")