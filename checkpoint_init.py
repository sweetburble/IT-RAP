import torch
from torch.optim import Adam
from stargan_solver import RainbowDQNNet  # This is defined by the user

# Configuration values must match the current config being used
state_dim = 128  # In case of being based on Mesonet
action_dim = 4
atom_size = 11
v_min = -5
v_max = 5
support = torch.linspace(v_min, v_max, atom_size)

# Initialize model and optimizer
model = RainbowDQNNet(state_dim, action_dim, atom_size, support)
optimizer = Adam(model.parameters(), lr=0.0001)

# Configure checkpoint
checkpoint = {
    'rainbow_dqn_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}
# Save
save_path = "stargan_celeba_256/models/final_rainbow_dqn.pth"
torch.save(checkpoint, save_path)
print(f"Initial Rainbow DQN weights saved successfully: {save_path}")