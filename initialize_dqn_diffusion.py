import os
import torch
from torch.optim import Adam
from diffusionclip_solver import RainbowDQNNet

# setting 
# if mesonet, state_dim should be 128, if Ghostfacenets, state_dim should be 1024
state_dim = 1024 
action_dim = 6
atom_size = 11
v_min = -5
v_max = 5
support = torch.linspace(v_min, v_max, atom_size)

# model & optimizer initialization
model = RainbowDQNNet(state_dim, action_dim, atom_size, support)
optimizer = Adam(model.parameters(), lr=0.0001)

# checkpoint setting
checkpoint = {
    'rainbow_dqn_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'meta': {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'atom_size': atom_size,
        'v_min': v_min,
        'v_max': v_max,
    },
}

# save
save_path = "checkpoints/models/diffusionclip/final_rainbow_dqn.pth"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(checkpoint, save_path)
print(f"RL model initial weights saved: {save_path}")