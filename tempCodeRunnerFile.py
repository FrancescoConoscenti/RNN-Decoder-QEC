class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, layers_sizes, hidden_size):
        super(FullyConnectedNN, self).__init__()
        
        layers = []

        layers.append(nn.Linear(input_size, hidden_size))

        """layers.append(nn.Linear(input_size, layers_sizes[0]))
        
        # Define hidden layers
        for i in range(len(layers_sizes) - 1):
            layers.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1]))
            layers.append(nn.ReLU())
        
        # Define output layer
        layers.append(nn.Linear(layers_sizes[-1], hidden_size))"""

        # Define activation function (e.g., ReLU)
        layers.append(nn.ReLU())

        # Combined sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)