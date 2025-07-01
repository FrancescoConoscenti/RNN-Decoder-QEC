import stim
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.multiprocessing as mp

class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        """
        Multi-layer fully connected network
        
        Args:
            input_size: Number of input features
            hidden_layers: List of hidden layer sizes
            output_size: Size of output
        """
        super(FullyConnectedNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers with activations
        for size in hidden_layers:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
            
        # Add output layer (no activation - handled separately)
        layers.append(nn.Linear(prev_size, output_size))
        
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
        
        # Process combined hidden states (horizontal, vertical and previous)
        #self.hidden_processor = FullyConnectedNN(hidden_size*3, fc_layers, hidden_size)
        self.hidden_processor = nn.Linear(hidden_size*3, hidden_size)
        #self.cell_processor = FullyConnectedNN(hidden_size*3, fc_layers, hidden_size)
        self.cell_processor = nn.Linear(hidden_size*3, hidden_size)
        
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
        device = x.device
        hidden_left, cell_left, hidden_up, cell_up, hidden_prev, cell_prev = hidden_states
        
        # Initialize missing hidden states with zeros if needed
        if hidden_left is None:
            hidden_left = torch.zeros(self.batch_size, self.hidden_size, device=device)
            cell_left = torch.zeros(self.batch_size, self.hidden_size, device=device)
        if hidden_up is None:
            hidden_up = torch.zeros(self.batch_size, self.hidden_size, device=device)
            cell_up = torch.zeros(self.batch_size, self.hidden_size, device=device)
            
        # Combine hidden states from different directions
        combined_h = torch.cat((hidden_left, hidden_up, hidden_prev), dim=1)
        combined_c = torch.cat((cell_left, cell_up, cell_prev), dim=1)
        
        # Process combined hidden states
        processed_h = self.hidden_processor(combined_h)
        processed_c = self.cell_processor(combined_c)
        
        # Update hidden state using LSTM cell
        x = x.squeeze(1).float()
        hidden, cell = self.lstm_cell(x, (processed_h, processed_c))
        
        return hidden, cell

class LatticeRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, grid_height, grid_width, fc_layers_intra, fc_layers_out, batch_size):
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
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.bn =  nn.BatchNorm1d(output_size)
        
        # Create a grid of RNN cells
        self.cells = nn.ModuleList([
            LatticeRNNCell(input_size, hidden_size, fc_layers_intra, batch_size) 
            for _ in range(grid_height * grid_width)
        ])
        
        # Output layer
        #self.fc_out = nn.Linear(hidden_size, output_size)
        self.fc_out = FullyConnectedNN(hidden_size * 2, fc_layers_out, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, h_ext, c_ext, grid_states):
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
       
        #x = x.reshape(batch_size,self.grid_height, self.grid_width)
        
        # Process each cell in the grid
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                # Get input for current cell
                cell_input = x[:, i, j].unsqueeze(1).unsqueeze(1)
                
                # Get previous states for this cell
                h_prev, c_prev = grid_states[i][j]
                
                # Handle special case for the first cell
                if i == 0 and j == 0:
                    h_left = h_ext
                    c_left = c_ext
                # Get left neighbor hidden state
                elif j > 0:
                    h_left, c_left = grid_states[i][j-1]
                else:
                    h_left, c_left = None, None
                
                # Get upper neighbor hidden state
                if i > 0:
                    h_up, c_up = grid_states[i-1][j]
                else:
                    h_up, c_up = None, None
                
                # Get cell index and process
                cell_index = i * self.grid_width + j
                h_new, c_new = self.cells[cell_index](
                    cell_input, 
                    (h_left, c_left, h_up, c_up, h_prev, c_prev)
                )
                
                # Update grid state
                grid_states[i][j] = (h_new, c_new)
        
        # Get final hidden state from bottom-right corner
        final_h, final_c = grid_states[-1][-1]

        # skip connection
        final_h = final_h + h_ext   
        final_c = final_c + c_ext
        
        final = torch.cat((final_h, final_c), dim=1)

        # Generate output
        output = self.fc_out(final)
        output = self.bn(output)
        output = self.sigmoid(output)
        
        
        return output, final_h, final_c, grid_states

class BlockRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, grid_height, grid_width, fc_layers_intra, fc_layers_out, batch_size):
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
        self.rnn_block = LatticeRNN(
            input_size, hidden_size, output_size, 
            grid_height, grid_width, fc_layers_intra, fc_layers_out, batch_size
        )
    
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
        grid_states = [
            [(h_ext, c_ext) for _ in range(grid_width)] 
            for _ in range(grid_height)
        ]
        
        # Process each round
        for round_idx in range(num_rounds):
            # Get input for this round
            round_input = x[:, round_idx,:]
            
            # Process through lattice RNN
            output, h_ext, c_ext, grid_states = self.rnn_block(
                round_input, h_ext, c_ext, grid_states
            )
        
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
        shuffle=False, drop_last=True
    )
    
    return train_loader, test_loader, X_train, X_test, y_train, y_test

def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs, num_rounds, device='cuda'):
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
        scheduler.step(running_loss)
        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{num_epochs}], lr: {current_lr} , Loss: {avg_loss:.4f}")
    
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
    model.to(device)  # Ensure model is on the correct device
    model.eval()
    correct = 0
    total = 0
    predictions = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            # Move data to device - this was missing!
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

def load_data(num_shots):
    """
    Load data from a .npz file
    
    Args:
        file_path: Path to the .npz file
        num_shots: Number of shots to load
        
    Returns:
        detection_array: Array of detection events
        observable_flips: Array of observable flips
    """
    # Load the compressed data
    if rounds == 5:
        loaded_data = np.load('data_stim/google_r5.npz')
    if rounds == 11:
        loaded_data = np.load('data_stim/google_r11.npz')
    if rounds == 17:
        loaded_data = np.load('data_stim/google_r17.npz')

    detection_array1 = loaded_data['detection_array1']
    detection_array1 = detection_array1[0:num_shots,:,:]
    observable_flips = loaded_data['observable_flips']
    observable_flips = observable_flips[0:num_shots]
        
    return detection_array1, observable_flips



# Configuration parameters
distance = 3
rounds = 11
num_shots = 50000

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



"""path = r"google_qec3v5_experiment_data/surface_code_bX_d3_r05_center_3_5/circuit_noisy.stim"
circuit_google = stim.Circuit.from_file(path)

# Compile the sampler
sampler = circuit_google.compile_detector_sampler()
# Sample shots, with observables
detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

detection_events = detection_events.astype(int)
detection_strings = [''.join(map(str, row)) for row in detection_events] #compress the detection events in a tensor
detection_events_numeric = [[int(value) for value in row] for row in detection_events] # Convert string elements to integers (or floats if needed)
detection_array = np.array(detection_events_numeric) # Convert detection_events to a numpy array
detection_array1 = detection_array.reshape(num_shots, rounds, num_ancilla_qubits) #first dim is the number of shots, second dim round number, third dim is the Ancilla"""


# Load data
detection_array1, observable_flips = load_data(num_shots)

order = [0,5,1,3,4,6,2,7] # Reorder using advanced indexing to create the chain connectivity
detection_array_ordered = detection_array1[..., order]
observable_flips = observable_flips.astype(int).flatten().tolist()

# Model hyperparameters
input_size = 1
hidden_size = 128
output_size = 1
grid_height = 4
grid_width = 2
batch_size = 256
test_size = 0.2
learning_rate = 0.001
num_epochs = 60
fc_layers_intra = [0] #not used
fc_layers_out = [hidden_size//2]

# Print configuration
print(f"2D LSTM")
print(f"Configuration: rounds={rounds}, distance={distance}, num_shots={num_shots}")
print(f"Model parameters: hidden_size={hidden_size}, batch_size={batch_size}")
print(f"Training parameters: learning_rate={learning_rate}, num_epochs={num_epochs}")

#adapt input data topology of model
detection_array_2D = detection_array_ordered.reshape(num_shots, rounds, grid_height, grid_width)

# Create data loaders
train_loader, test_loader, X_train, X_test, y_train, y_test = create_data_loaders(
detection_array_2D, observable_flips, batch_size, test_size
)

# Create model
model = BlockRNN(input_size, hidden_size, output_size, grid_height, 
                 grid_width, fc_layers_intra, fc_layers_out, batch_size).to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)


# Train model
model, losses = train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs, rounds, device)

# Evaluate model
accuracy, predictions = evaluate_model(model, test_loader, rounds, device)

# Save model
torch.save(model.state_dict(), "2D_LSTM_r11.pth")
print(f"Model saved to 2D_LSTM_r11.pth")