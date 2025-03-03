import stim
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, layers_sizes, hidden_size):
        super(FullyConnectedNN, self).__init__()
        
        layers = []

        layers.append(nn.Linear(input_size, layers_sizes[0]))
        
        # Define hidden layers
        for i in range(len(layers_sizes) - 1):
            layers.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1]))
            layers.append(nn.ReLU())
        
        # Define output layer
        layers.append(nn.Linear(layers_sizes[-1], hidden_size))

        # Define activation function (e.g., ReLU)
        layers.append(nn.ReLU())

        # Combined sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class LatticeRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, fc_layers, batch_size):
        """
        Custom RNN cell that processes inputs in a 2D lattice structure
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            fc_layers: List of layer sizes for fully connected networks
            batch_size: Batch size for training
        """
        super(LatticeRNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        
        # Process input
        self.fc_input = nn.Linear(input_size, input_size)
        
        # Process combined hidden states 
        # (precedent chain element and previous in time so input dim = hidden_size*2)
        self.hidden_processor = FullyConnectedNN(hidden_size*2, fc_layers, hidden_size)
        self.cell_processor = FullyConnectedNN(hidden_size*2, fc_layers, hidden_size)
        
        # LSTM cell for time dimension
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        
    def forward(self, x, hidden_states):
        """
        Forward pass for the lattice RNN cell
        
        Args:
            x: Input tensor [batch_size, input_size]
            hidden_states: Tuple containing:
                - hidden_left: Hidden state from left neighbor
                - hidden_up: Hidden state from upper neighbor
                - hidden_prev: Previous hidden state
                - cell_prev: Previous cell state
                
        Returns:
            Tuple of (hidden_state, cell_state)
        """
        hidden, cell, hidden_prev, cell_prev = hidden_states
        device = x.device
        
        # Convert to tensors if necessary
        hidden_prev = hidden_prev.to(device)
        cell_prev = cell_prev.to(device)
        
        # Initialize missing hidden states with zeros if needed
        if hidden is None:
            hidden = torch.zeros(self.batch_size, self.hidden_size, device=device)
            cell = torch.zeros(self.batch_size, self.hidden_size, device=device)
            
        # Combine hidden states from different directions
        combined_h = torch.cat((hidden, hidden_prev), dim=1)
        combined_c = torch.cat((cell, cell_prev), dim=1)
        
        # Process combined hidden states
        processed_h = self.hidden_processor(combined_h)
        processed_c = self.cell_processor(combined_c)
        
        # Update hidden state using LSTM cell
        x = x.squeeze(1).float()
        hidden, cell = self.lstm_cell(x, (processed_h, processed_c))
        
        return hidden, cell

class LatticeRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, length, fc_layers, batch_size):
        """
        Network that processes inputs in a 2D lattice structure
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            output_size: Size of output
            grid_height: Height of the 2D grid
            grid_width: Width of the 2D grid
            fc_layers: List of layer sizes for fully connected networks
            batch_size: Batch size for training
        """
        super(LatticeRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.chain_length = length
        
        # Create a grid of RNN cells
        self.cells = nn.ModuleList([
            LatticeRNNCell(input_size, hidden_size, fc_layers, batch_size) 
            for _ in range(chain_length)
        ])
        
        # Output layer
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, h_ext, c_ext, chain_states):
        """
        Forward pass for the lattice RNN
        
        Args:
            x: Input tensor [batch_size, grid_height, grid_width]
            h_ext: External hidden state
            c_ext: External cell state
            grid_states: Previous states for the grid
            
        Returns:
            output: Output tensor
            final_h: Final hidden state
            final_c: Final cell state
            grid_states: Updated grid states
        """
        batch_size = x.size(0)
        device = x.device
        x = x.squeeze(2)
        
        # Process each cell in the grid
        for i in range(self.chain_length):
            
            # Get input for current cell
            cell_input = x[:, i].unsqueeze(1).unsqueeze(1)
                
            # Get previous states for this cell
            h_prev, c_prev = chain_states[i]
                
            # Handle special case for the first cell
            if i == 0:
                h = h_ext
                c = c_ext   
            # Get upper neighbor hidden state
            else:
                h, c = chain_states[i-1]
            
            # Get cell index and process
            h_new, c_new = self.cells[i](cell_input, (h, c, h_prev, c_prev))
                
            # Update grid state
            chain_states[i] = (h_new, c_new)
        
        # Get final hidden state from bottom-right corner
        final_h, final_c = chain_states[-1]
        
        # Generate output
        output = self.fc_out(final_h)
        output = self.sigmoid(output)
        
        return output, final_h, final_c, chain_states

class BlockRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, chain_length, fc_layers, batch_size):
        """
        Block RNN model that processes multiple time steps of data on a 2D lattice
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            output_size: Size of output
            grid_height: Height of the 2D grid
            grid_width: Width of the 2D grid
            fc_layers: List of layer sizes for fully connected networks
            batch_size: Batch size for training
        """
        super(BlockRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        
        # Input processing
        self.fc_in = nn.Linear(input_size, input_size)
        
        # Lattice RNN for spatial processing
        self.rnn_block = LatticeRNN(input_size, hidden_size, output_size, chain_length, fc_layers, batch_size)
    
    def forward(self, x, num_rounds):
        """
        Forward pass for the Block RNN
        
        Args:
            x: Input tensor [batch_size, num_rounds, grid_size]
            num_rounds: Number of time steps to process
            
        Returns:
            output: Final prediction
            final_h: Final hidden state
        """
        batch_size = x.size(0)
        device = x.device
        
        # Initialize external hidden states
        h_ext = torch.zeros(batch_size, self.hidden_size, device=device)
        c_ext = torch.zeros(batch_size, self.hidden_size, device=device)
        
        # Initialize grid states
        chain_states = [(h_ext, c_ext) for _ in range(chain_length)] 
            
        
        # Process each round
        for round_idx in range(num_rounds):
            # Get input for this round
            round_input = x[:, round_idx,:].unsqueeze(2)
            
            # Process through lattice RNN
            output, h_ext, c_ext, chain_states = self.rnn_block(round_input, h_ext, c_ext, chain_states)
        
        return output, h_ext

def create_data_loaders(detection_array, observable_flips, batch_size, test_size=0.2):
    """
    Create PyTorch DataLoaders for training and testing
    
    Args:
        detection_array: Array of detection events
        observable_flips: Array of observable flips
        batch_size: Batch size for training
        test_size: Fraction of data to use for testing
        
    Returns:
        train_loader: DataLoader for training
        test_loader: DataLoader for testing
        X_train: Training data
        X_test: Testing data
        y_train: Training labels
        y_test: Testing labels
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        detection_array, observable_flips, 
        test_size=test_size, shuffle=False
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_train_tensor = torch.tensor(y_train).float()
    y_test_tensor = torch.tensor(y_test).float()
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=False, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, 
        shuffle=False, drop_last=False
    )
    
    return train_loader, test_loader, X_train, X_test, y_train, y_test

def train_model(model, train_loader, criterion, optimizer, num_epochs, num_rounds, device='cuda'):
    """
    Train the model
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train for
        num_rounds: Number of rounds in the data
        device: Device to train on ('cuda' or 'cpu')
        
    Returns:
        model: Trained model
        losses: List of losses per epoch
    """
    model.to(device)
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            # Move data to device
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output, _ = model(batch_x, num_rounds)
            loss = criterion(output.squeeze(1), batch_y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Calculate average loss for this epoch
        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    print("Training finished.")
    return model, losses

def evaluate_model(model, test_loader, num_rounds, device='cuda'):
    """
    Evaluate the model on test data
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader for test data
        num_rounds: Number of rounds in the data
        device: Device to evaluate on ('cuda' or 'cpu')
        
    Returns:
        accuracy: Test accuracy
        predictions: Model predictions
    """
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    predictions = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            # Move data to device
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            output, _ = model(batch_x, num_rounds)
            
            # Get predictions
            predicted = (output.squeeze(1) > 0.5).float()
            predictions.extend(predicted.cpu().numpy())
            
            # Calculate accuracy
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
    
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    
    return accuracy, predictions

def load_data(file_path, num_shots):
    """
    Load data from a .npz file
    
    Args:
        file_path: Path to the .npz file
        num_shots: Number of shots to load
        
    Returns:
        detection_array: Array of detection events
        observable_flips: Array of observable flips
    """
    loaded_data = np.load(file_path)
    detection_array = loaded_data['detection_array1'][:num_shots, :, :]
    observable_flips = loaded_data['observable_flips'][:num_shots]
    
    return detection_array, observable_flips


# Configuration parameters
distance = 3
rounds = 5
num_shots = 1000

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Determine system size based on distance
if distance == 3:
    num_qubits = 17
    num_data_qubits = 9
    num_ancilla_qubits = 8
elif distance == 5:
    num_qubits = 49
    num_data_qubits = 25
    num_ancilla_qubits = 24

"""# Load data
data_path = 'data_stim/google_r5.npz'
detection_array, observable_flips = load_data(data_path, num_shots)"""


path = r"google_qec3v5_experiment_data/surface_code_bX_d3_r05_center_3_5/circuit_noisy.stim"
circuit_google = stim.Circuit.from_file(path)

# Compile the sampler
sampler = circuit_google.compile_detector_sampler()
# Sample shots, with observables
detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

detection_events = detection_events.astype(int)
detection_strings = [''.join(map(str, row)) for row in detection_events] #compress the detection events in a tensor
detection_events_numeric = [[int(value) for value in row] for row in detection_events] # Convert string elements to integers (or floats if needed)
detection_array = np.array(detection_events_numeric) # Convert detection_events to a numpy array

detection_array1 = detection_array.reshape(num_shots, rounds, num_ancilla_qubits) #first dim is the number of shots, second dim round number, third dim is the Ancilla 

observable_flips = observable_flips.astype(int).flatten().tolist()

# Model hyperparameters
input_size = 1
hidden_size = 128
output_size = 1
chain_length = num_ancilla_qubits
batch_size = 64
test_size = 0.2
learning_rate = 0.005
num_epochs = 1
fc_layers = [hidden_size*3, hidden_size*2, hidden_size]

# Print configuration
print(f"2D LSTM connection")
print(f"Configuration: rounds={rounds}, distance={distance}, num_shots={num_shots}")
print(f"Model parameters: hidden_size={hidden_size}, batch_size={batch_size}")
print(f"Training parameters: learning_rate={learning_rate}, num_epochs={num_epochs}")

# Create data loaders
train_loader, test_loader, X_train, X_test, y_train, y_test = create_data_loaders(
detection_array1, observable_flips, batch_size, test_size
)

# Create model
model = BlockRNN(input_size, hidden_size, output_size, chain_length, fc_layers, batch_size).to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train model
model, losses = train_model(model, train_loader, criterion, optimizer, num_epochs, rounds, device)

# Evaluate model
accuracy, predictions = evaluate_model(model, test_loader, rounds, device)

# Save model
torch.save(model.state_dict(), "2D_LSTM_r11.pth")
print(f"Model saved to 2D_LSTM_r11.pth")