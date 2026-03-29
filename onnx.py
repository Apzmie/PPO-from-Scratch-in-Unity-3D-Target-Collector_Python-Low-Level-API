import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mean = torch.tanh(self.mean(x))
        std = self.log_std.exp().expand_as(mean)
        value = self.critic(x).squeeze(-1)
        return mean, std, value
        
state_dim = 11
action_dim = 2

model = ActorCritic(state_dim, action_dim)
state_dict = torch.load("saved_model.pth", weights_only=True)
model.load_state_dict(state_dict)
model.eval()

dummy_input = torch.randn(1, state_dim)

torch.onnx.export(
    model,
    dummy_input,
    "saved_model.onnx",
    export_params=True,
    opset_version=15,
    do_constant_folding=True,
    input_names=['X'],
    output_names=['Y'],
)
