import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh(),
            nn.Linear(8, 16),
            nn.Tanh(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_weights(self):
        return {name: param.data.cpu().numpy() for name, param in self.state_dict().items()}

    def set_weights(self, weights):
        state_dict = self.state_dict()
        for name, weight in weights.items():
            if name in state_dict:
                state_dict[name].copy_(torch.tensor(weight))
        self.load_state_dict(state_dict)
