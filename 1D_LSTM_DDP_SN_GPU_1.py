import stim
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
import time

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
        
        """# Initialize missing hidden states with zeros if needed
        if hidden is None:
            hidden = torch.zeros(self.batch_size, self.hidden_size, device=device)
            cell = torch.zeros(self.batch_size, self.hidden_size, device=device)"""
            
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
    def __init__(self, input_size, hidden_size, output_size, length, fc_layers_intra, fc_layers_out, batch_size):
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
            LatticeRNNCell(input_size, hidden_size, fc_layers_intra, batch_size) 
            for _ in range(self.chain_length)
        ])
        
        # Output layer
        self.fc_out = FullyConnectedNN(hidden_size*2, fc_layers_out, output_size)
        self.bn = nn.BatchNorm1d(output_size)
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
                
            #chain:states[i] has the h,c of the previous round, 
            #continuing the loop I overwrite element of chain_states with the h,c spatial
            h_time, c_time= chain_states[i]
                
            # Handle special case for the first cell
            if i == 0:
                h_space = h_ext
                c_space = c_ext   
            # Get spacial neighbor hidden state, from the previous LatticeRNNCell in space
            else:
                h_space, c_space = chain_states[i-1]
            
            # Get cell index and process
            h_new, c_new = self.cells[i](cell_input, (h_space, c_space, h_time, c_time))
                
            # Update grid state
            chain_states[i] = (h_new, c_new)
        
        # Get final hidden state from bottom-right corner
        final_h, final_c = chain_states[-1]
        
        final = torch.cat((final_h, final_c), dim=1)
        
        # Generate output
        output = self.fc_out(final)
        output = self.bn(output)
        output = self.sigmoid(output)
        
        return output, final_h, final_c, chain_states

class BlockRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, chain_length, fc_layers_intra, fc_layers_out, batch_size):
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
        self.chain_length = chain_length
        
        # Input processing
        self.fc_in = nn.Linear(input_size, input_size)
        
        # Lattice RNN for spatial processing
        self.rnn_block = LatticeRNN(input_size, hidden_size, output_size, chain_length, fc_layers_intra,fc_layers_out, batch_size)
    
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
        chain_states = [(h_ext, c_ext) for _ in range(self.chain_length)] 
            
        
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

    # Initialize DDP first (call this before creating loaders)
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

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
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True  # Shuffle per epoch
    ) if world_size > 1 else None
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, pin_memory=True,
        shuffle=False, drop_last=True, sampler=train_sampler)
    
    #I test only on the data in the first process
    if rank == 0:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        test_loader = None
    
    
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


def train_model(device, model, train_loader, criterion, optimizer, num_epochs, rounds):
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

    #model = model()#.to(rank)
    #model = DDP(model, find_unused_parameters=True, device_ids=[rank])


    losses = []
    
    for epoch in range(num_epochs):
        
        print(f"[{dist.get_rank()}] " + "-" * 10 + f" epoch {epoch + 1}/{5}")

        running_loss = 0.0
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        for batch_x, batch_y in train_loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output, _ = model(batch_x, rounds)
            loss = criterion(output.squeeze(1), batch_y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Calculate average loss for this epoch
        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)

        print(f"[{dist.get_rank()}] " + f"{epoch}/{num_epochs}, train_loss: {loss.item():.4f}")
    
    print("Training finished.")


    dist.destroy_process_group()

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

def load_data(num_shots,rounds):
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

def everything(rank, world_size):

    # Configuration parameters
    distance = 3
    rounds = 5
    num_shots = 100000

    # Determine system size based on distance
    if distance == 3:
        num_qubits = 17
        num_data_qubits = 9
        num_ancilla_qubits = 8
    elif distance == 5:
        num_qubits = 49
        num_data_qubits = 25
        num_ancilla_qubits = 24

    #Load data form compressed file .npz
    detection_array1, observable_flips = load_data(num_shots,rounds)

    # Reorder using advanced indexing to create the chain connectivity
    order = [0,3,5,6,7,4,2,1]
    detection_array_ordered = detection_array1[..., order]

    # Model hyperparameters
    input_size = 1
    hidden_size = 128
    output_size = 1
    chain_length = num_ancilla_qubits
    batch_size = 256
    test_size = 0.2
    learning_rate = 0.001
    num_epochs = 1
    fc_layers_intra = [0]
    fc_layers_out = [int(hidden_size/8)]

    # Print configuration
    print(f"1D LSTM DDP")
    print(f"Configuration: rounds={rounds}, distance={distance}, num_shots={num_shots}")
    print(f"Model parameters: hidden_size={hidden_size}, batch_size={batch_size}")
    print(f"Training parameters: learning_rate={learning_rate}, num_epochs={num_epochs}")


    device = torch.device(f'cuda:{rank}')     
    torch.cuda.set_device(device)

    # Create data loaders
    train_loader, test_loader, X_train, X_test, y_train, y_test = create_data_loaders(
    detection_array_ordered, observable_flips, batch_size, test_size)


    # Create model
    model = BlockRNN(input_size, hidden_size, output_size, chain_length, fc_layers_intra, 
                     fc_layers_out, batch_size).to(device)
    ddp_model = DDP(model,find_unused_parameters=True, device_ids=[device])# if torch.cuda.is_available() else None)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=learning_rate)

    start_time = time.time()
    train_model(device, model, train_loader, criterion, optimizer, num_epochs, rounds)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.6f} seconds")

    # Evaluate model
    accuracy, predictions = evaluate_model(device, ddp_model.module, test_loader, rounds)


def main():

    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS")))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID")))
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID')))
    
    ddp_setup(rank, world_size)

    everything(rank, world_size)
    
if __name__ == "__main__":
    main()