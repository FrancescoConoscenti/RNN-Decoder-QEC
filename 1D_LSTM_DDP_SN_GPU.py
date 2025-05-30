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
from typing import List
from torch.optim import AdamW
import torch.nn.init as init


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.LSTMCell):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                init.zeros_(param.data)


class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, layers_sizes, hidden_size, dropout_prob):
        super(FullyConnectedNN, self).__init__()
        
        layers = []

        if layers_sizes == [0]:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))  
            # Define activation function (e.g., ReLU)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))  # Dropout after each hidden layer

        else:

            layers.append(nn.Linear(input_size, layers_sizes[0]))
            layers.append(nn.LayerNorm(layers_sizes[0])) 
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))  # Dropout after first layer
            # Define hidden layers
            for i in range(len(layers_sizes) - 1):
                layers.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1]))
                layers.append(nn.LayerNorm(layers_sizes[i + 1])) 
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_prob))  # Dropout after first layer
            
            # Define output layer
            layers.append(nn.Linear(layers_sizes[-1], hidden_size))
            layers.append(nn.LayerNorm(hidden_size))  


        # Combined sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LatticeRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, fc_layers, batch_size, dropout_prob):
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
        self.hidden_processor = FullyConnectedNN(hidden_size*2, fc_layers, hidden_size, dropout_prob)
        self.cell_processor = FullyConnectedNN(hidden_size*2, fc_layers, hidden_size, dropout_prob)
        
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
    def __init__(self, input_size, hidden_size, output_size, length, fc_layers_intra, fc_layers_out, batch_size, dropout_prob):
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
            LatticeRNNCell(input_size, hidden_size, fc_layers_intra, batch_size, dropout_prob) 
            for _ in range(self.chain_length)
        ])
        
        # Output layer
        self.fc_out = FullyConnectedNN(hidden_size*2, fc_layers_out, output_size, dropout_prob)
        self.input_residual_proj = nn.Linear(input_size * self.chain_length, hidden_size*2)
        #self.bn = nn.BatchNorm1d(output_size)
        #self.sigmoid = nn.Sigmoid()
    
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

        #skip connections
        x = x.squeeze(1).float()  # or reshape appropriately
        x_proj = self.input_residual_proj(x)

        final = final + x_proj
        
        # Generate output
        output = self.fc_out(final)
        #output = self.bn(output)
        #output = self.sigmoid(output)
        
        return output, final_h, final_c, chain_states

class BlockRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, chain_length, fc_layers_intra, fc_layers_out, batch_size, dropout_prob):
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
        self.rnn_block = LatticeRNN(input_size, hidden_size, output_size, chain_length,
                                     fc_layers_intra,fc_layers_out, batch_size, dropout_prob)
    
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
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, drop_last=True, sampler=train_sampler)
    
    #I test only on the data in the first process
    if rank == 0:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        test_loader = None
    
    
    return train_loader, test_loader, X_train, X_test, y_train, y_test, train_sampler

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


def train_model(rank, model, train_loader, train_sampler, criterion, optimizer, scheduler, num_epochs, rounds):
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
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        print(f"[GPU{rank}] | Epoch {epoch} ")

        running_loss = 0.0
        grad_sum = 0.0
        count = 0
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        for batch_x, batch_y in train_loader:

            batch_x = batch_x.to(rank)
            batch_y = batch_y.to(rank)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output, _ = model(batch_x, rounds)
            loss = criterion(output.squeeze(1), batch_y)
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
        
        # Step the scheduler once per epoch
        scheduler.step(running_loss)
        # Calculate average loss for this epoch
        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)

        #gradient logging
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_sum += param.grad.abs().mean().item()
                count += 1

        if rank == 0:
            ckp = model.module.state_dict()
            PATH = "checkpoint.pt"
            torch.save(ckp, PATH)
            print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")


        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch [{epoch+1}/{num_epochs}], Avg gradient:{grad_sum / count:.6f}, lr: {current_lr:.5f}, Loss: {avg_loss:.5f}")
    
    
    print("Training finished.")


    dist.destroy_process_group()
    torch.cuda.empty_cache()

    return model, losses

def finetune(rank, model, train_loader_exp, criterion, optimizer, num_epochs_fine, rounds):

    losses = []

    for epoch in range(num_epochs_fine):
        running_loss = 0.0
        
        for batch_x, batch_y in train_loader_exp:

            batch_x = batch_x.to(rank)
            batch_y = batch_y.to(rank)

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
        avg_loss = running_loss / len(train_loader_exp)
        losses.append(avg_loss)


        print(f"Epoch [{epoch+1}/{num_epochs_fine}], Loss: {avg_loss:.4f}")
    
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

def parse_b8(data: bytes, bits_per_shot: int) -> List[List[bool]]:
    shots = []
    bytes_per_shot = (bits_per_shot + 7) // 8
    for offset in range(0, len(data), bytes_per_shot):
        shot = []
        for k in range(bits_per_shot):
            byte = data[offset + k // 8]
            bit = (byte >> (k % 8)) % 2 == 1
            shot.append(bit)
        shots.append(shot)
    return shots

def load_data_exp():

    # Load the compressed data
    if rounds == 5:
        path1 = r"google_qec3v5_experiment_data/surface_code_bX_d3_r05_center_3_5/detection_events.b8"
        path2 = r"google_qec3v5_experiment_data/surface_code_bX_d3_r05_center_3_5/obs_flips_actual.01"
    if rounds == 11:
        path1 = r"google_qec3v5_experiment_data/surface_code_bX_d3_r11_center_3_5/detection_events.b8"
        path2 = r"google_qec3v5_experiment_data/surface_code_bX_d3_r11_center_3_5/obs_flips_actual.01"
    if rounds == 17:
        path1 = r"google_qec3v5_experiment_data/surface_code_bX_d3_r17_center_3_5/detection_events.b8"
        path2 = r"google_qec3v5_experiment_data/surface_code_bX_d3_r17_center_3_5/obs_flips_actual.01"

    bits_per_shot = rounds*8

    with open(path1, "rb") as file:
        # Read the file content as bytes
        data_detection = file.read()

    detection_exp = parse_b8(data_detection,bits_per_shot)
    detection_exp1 = np.array(detection_exp)
    detection_exp2 = detection_exp1.reshape(50000, rounds, num_ancilla_qubits)


    with open(path2, "rb") as file:
        # Read the file content as bytes
        data_obs = file.read()

    obs_exp = data_obs.replace(b"\n", b"")
    obs_exp_bit = [bit-48 for bit in obs_exp]
    obs_exp_bit_array = np.array(obs_exp_bit)

    X_train_exp, X_test_exp, y_train_exp, y_test_exp = train_test_split(detection_exp2, obs_exp_bit_array, test_size=0.2, random_state=42, shuffle=False)

    return detection_exp2, obs_exp_bit_array 

def main(rank, local_rank, train_param, dataset, Net_Arch, world_size):
    
    # Set GPU FIRST
    torch.cuda.set_device(rank)
    print(f"Initializing Rank {rank} on GPU {torch.cuda.current_device()}")

    ddp_setup(rank, world_size)
    
    # Verify communication
    tensor = torch.tensor([1.0]).cuda()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {rank} | Sum: {tensor.item()}") 

    num_epochs, num_epochs_fine, rounds, learning_rate, learning_rate_fine, batch_size, dropout_prob = train_param
    detection_array_ordered, observable_flips,  detection_array_ordered_exp, observable_flips_exp, test_size = dataset
    input_size, hidden_size, output_size, chain_length, fc_layers_intra, fc_layers_out = Net_Arch

    # Create data loaders
    train_loader, test_loader, X_train, X_test, y_train, y_test, train_sampler = create_data_loaders(
    detection_array_ordered, observable_flips, batch_size, test_size)

    #train_loader_exp, test_loader_exp, X_train_exp, X_test_exp, y_train_exp, y_test_exp, train_sampler = create_data_loaders(
    #detection_array_ordered_exp, observable_flips_exp, batch_size, test_size)

    torch.cuda.set_device(local_rank)
    gpu = torch.device("cuda")

    # Create model
    model = BlockRNN(input_size, hidden_size, output_size, chain_length, fc_layers_intra, 
                     fc_layers_out, batch_size, dropout_prob)
    model.apply(initialize_weights)
    model. to(gpu)
    ddp_model = DDP(model, find_unused_parameters=True, device_ids=[local_rank])# if torch.cuda.is_available() else None)

    #train
    # Define loss function and optimizer
    #criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    #optimizer = optim.Adam(ddp_model.parameters(), lr=learning_rate)

    train_model(rank, ddp_model, train_loader, train_sampler, criterion, optimizer, scheduler, num_epochs, rounds)

    #finetuning
    #optimizer = optim.Adam(ddp_model.parameters(), lr=learning_rate_fine)
    #finetune(rank, ddp_model.module, train_loader_exp, criterion, optimizer, num_epochs_fine, rounds)


    # Evaluate model
    accuracy, predictions = evaluate_model(rank, ddp_model.module, test_loader, rounds)
    #accuracy, predictions = evaluate_model(rank, ddp_model.module, test_loader_exp, rounds)



if __name__ == "__main__":
    #mp.set_start_method("spawn", force=True)  # Ensure safe multiprocessing
    #os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO' # Debug for unused gradients
        
    # Configuration parameters
    distance = 3
    rounds = 17
    num_shots = 1000000

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
    detection_array1, observable_flips = load_data(num_shots)
    detection_array_exp, observable_flips_exp = load_data_exp()

    # Reorder using advanced indexing to create the chain connectivity
    order = [0,3,5,6,7,4,2,1]
    detection_array_ordered = detection_array1[..., order]
    detection_array_ordered_exp = detection_array_exp[..., order]

    # Model hyperparameters
    input_size = 1
    hidden_size = 128
    output_size = 1
    chain_length = num_ancilla_qubits
    batch_size = 128
    test_size = 0.2
    learning_rate = 0.001
    learning_rate_fine = 0.0001
    dropout_prob = 0.1
    num_epochs = 20
    num_epochs_fine = 5
    fc_layers_intra = [0]
    fc_layers_out = [int(hidden_size/8)]

    # Print configuration
    print(f"1D LSTM DDP")
    print(f"Configuration: rounds={rounds}, distance={distance}, num_shots={num_shots}")
    print(f"Model parameters: hidden_size={hidden_size}, batch_size={batch_size}")
    print(f"Training parameters: learning_rate={learning_rate}, num_epochs={num_epochs}, dropout_prob={dropout_prob}")

    #world_size = torch.cuda.device_count()
    #world_size = int(os.environ.get("WORLD_SIZE", 1))  # Changed: Use environment variable

    
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS")))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID")))
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID')))

    start_time = time.time()
    
    # For SLURM launches
    main(rank=rank, local_rank=local_rank,
        train_param=(num_epochs, num_epochs_fine, rounds, learning_rate, learning_rate_fine, batch_size, dropout_prob),
        dataset=(detection_array_ordered, observable_flips, detection_array_ordered_exp, observable_flips_exp, test_size),
        Net_Arch=(input_size, hidden_size, output_size, chain_length, fc_layers_intra, fc_layers_out),
        world_size=world_size)

    end_time = time.time()

    # Print execution time
    print(f"Execution time: {end_time - start_time:.6f} seconds")

    # Save model
    #torch.save(model.state_dict(), "2D_LSTM_r11.pth")
    #print(f"Model saved to 2D_LSTM_r11.pth")
