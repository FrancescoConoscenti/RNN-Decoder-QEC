import stim
import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.multiprocessing as mp
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, layers_sizes, hidden_size):
        super(FullyConnectedNN, self).__init__()
        
        layers = []

        if layers_sizes == [0]:
            layers.append(nn.Linear(input_size, hidden_size))
            # Define activation function (e.g., ReLU)
            layers.append(nn.ReLU())

        else:
            layers.append(nn.Linear(input_size, layers_sizes[0]))
            # Define hidden layers
            for i in range(len(layers_sizes) - 1):
                layers.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1]))
                layers.append(nn.ReLU())
            
            # Define output layer
            layers.append(nn.Linear(layers_sizes[-1], hidden_size))


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
        hidden_left, cell_left, hidden_up, cell_up, hidden_prev, cell_prev = hidden_states
        device = x.device
        
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
        
        # Create a grid of RNN cells
        self.cells = nn.ModuleList([
            LatticeRNNCell(input_size, hidden_size, fc_layers_intra, batch_size) 
            for _ in range(grid_height * grid_width)
        ])
        
        # Output layer
        #self.fc_out = nn.Linear(hidden_size, output_size)
        self.fc_out = FullyConnectedNN(hidden_size, fc_layers_out, output_size)
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
        
        # Generate output
        output = self.fc_out(final_h)
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

def ddp_setup(rank, world_size):
    
    #Args:
    #    rank: Unique identifier of each process
    #    world_size: Total number of processes
    
    # Use environment variables that are already set by SLURM script
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12355"


    dist.init_process_group(
        backend="nccl",  # Must use "gloo" for CPU
        rank=rank,
        world_size=world_size
    )


def train_model(rank, model, train_loader, criterion, optimizer, num_epochs, num_rounds, device='cuda'):
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

def evaluate_model(rank, model, test_loader, num_rounds):
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

    model.eval()
    correct = 0
    total = 0
    predictions = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:

            batch_x = batch_x.to(rank)
            batch_y = batch_y.to(rank)
            
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

def load_data(num_shots, rounds):
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

def main(rank, local_rank, train_param, dataset, Net_Arch, world_size):
    torch.cuda.set_device(rank)
    print(f"Initializing Rank {rank} on GPU {torch.cuda.current_device()}")

    ddp_setup(rank, world_size)

    # Test DDP communication
    tensor = torch.tensor([1.0]).cuda()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {rank} | Sum: {tensor.item()}")

    # Unpack parameters
    num_epochs, rounds, learning_rate, batch_size = train_param
    detection_array_ordered, observable_flips, test_size = dataset
    input_size, hidden_size, output_size, grid_height, grid_width, fc_layers_intra, fc_layers_out = Net_Arch

    # Adapt data topology
    detection_array_2D = detection_array_ordered.reshape(-1, rounds, grid_height, grid_width)

    # Data loaders
    train_loader, test_loader, X_train, X_test, y_train, y_test = create_data_loaders(
        detection_array_2D, observable_flips, batch_size, test_size)

    # Model
    gpu = torch.device("cuda")
    model = BlockRNN(input_size, hidden_size, output_size, grid_height, grid_width,
                     fc_layers_intra, fc_layers_out, batch_size).to(gpu)
    ddp_model = DDP(model, find_unused_parameters=True, device_ids=[local_rank])

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=learning_rate)

    # Train
    train_model(rank, ddp_model, train_loader, criterion, optimizer, num_epochs, rounds)

    # Evaluate
    accuracy, predictions = evaluate_model(rank, ddp_model.module, test_loader, rounds)


if __name__ == "__main__":
    import numpy as np

    # Configuration
    distance = 3
    rounds = 5
    num_shots = 5000

    if distance == 3:
        num_qubits = 17
        num_data_qubits = 9
        num_ancilla_qubits = 8
    elif distance == 5:
        num_qubits = 49
        num_data_qubits = 25
        num_ancilla_qubits = 24

    # Load and reorder data
    detection_array1, observable_flips = load_data(num_shots,rounds)
    order = [0, 5, 1, 3, 4, 6, 2, 7]
    detection_array_ordered = detection_array1[..., order]
    observable_flips = np.array(observable_flips).astype(int).flatten().tolist()

    # Model hyperparameters
    input_size = 1
    hidden_size = 64
    output_size = 1
    grid_height = 4
    grid_width = 2
    batch_size = 256
    test_size = 0.2
    learning_rate = 0.002
    num_epochs = 1
    fc_layers_intra = [0]
    fc_layers_out = [hidden_size]

    print(f"2D LSTM DDP")
    print(f"Configuration: rounds={rounds}, distance={distance}, num_shots={num_shots}")
    print(f"Model parameters: hidden_size={hidden_size}, batch_size={batch_size}")
    print(f"Training parameters: learning_rate={learning_rate}, num_epochs={num_epochs}")

    # World size and rank
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))

    start_time = time.time()

    # Run main function
    main(rank=rank, local_rank=local_rank,
         train_param=(num_epochs, rounds, learning_rate, batch_size),
         dataset=(detection_array_ordered, observable_flips, test_size),
         Net_Arch=(input_size, hidden_size, output_size, grid_height, grid_width,
                   fc_layers_intra, fc_layers_out),
         world_size=world_size)

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.6f} seconds")

