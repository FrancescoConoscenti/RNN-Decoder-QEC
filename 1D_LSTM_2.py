import stim
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
import time
from typing import List

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

def count_parameters(model):
    """Count the total number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_data_loaders(detection_array, observable_flips, batch_size, test_size=0.2, val_size=0.2):
    """
    Create PyTorch DataLoaders for training, validation, and testing
    
    Args:
        detection_array: Array of detection events
        observable_flips: Array of observable flips
        batch_size: Batch size for training
        test_size: Fraction of data to use for testing
        val_size: Fraction of remaining data to use for validation
        
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        test_loader: DataLoader for testing
        X_train, X_val, X_test: Data splits
        y_train, y_val, y_test: Label splits
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        detection_array, observable_flips, 
        test_size=test_size, shuffle=False
    )
    
    # Second split: separate train and validation from remaining data
    val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted, shuffle=False
    )
     
    # Convert to PyTorch tensors if needed
    if isinstance(detection_array, (np.ndarray, list)):
        X_train = torch.from_numpy(X_train).float()
        X_val = torch.from_numpy(X_val).float()
        X_test = torch.from_numpy(X_test).float()

    if isinstance(observable_flips, (np.ndarray, list)):
        y_train = torch.tensor(y_train).float()
        y_val = torch.tensor(y_val).float()
        y_test = torch.tensor(y_test).float()
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=False, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, 
        shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, 
        shuffle=False, drop_last=False
    )
    
    return train_loader, val_loader, test_loader, X_train, X_val, X_test, y_train, y_val, y_test

def evaluate_model_validation(model, data_loader, num_rounds):
    """
    Evaluate the model and return loss and accuracy
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader for evaluation data
        num_rounds: Number of rounds in the data
        
    Returns:
        avg_loss: Average loss
        accuracy: Accuracy
    """
    model.eval()
    criterion = nn.BCELoss()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            # Forward pass
            output, _ = model(batch_x, num_rounds)
            loss = criterion(output.squeeze(1), batch_y)
            total_loss += loss.item()
            
            # Get predictions
            predicted = (output.squeeze(1) > 0.5).float()
            
            # Calculate accuracy
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, num_rounds, scheduler=None):
    """
    Train the model with validation
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train for
        num_rounds: Number of rounds in the data
        scheduler: Learning rate scheduler (optional)
        
    Returns:
        model: Trained model
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        val_accuracies: List of validation accuracies per epoch
    """
    model.train()
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output, _ = model(batch_x, num_rounds)
            loss = criterion(output.squeeze(1), batch_y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Calculate average training loss for this epoch
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_loss, val_acc = evaluate_model_validation(model, val_loader, num_rounds)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch results
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"LR: {current_lr}")
        
        # Step the scheduler if provided
        if scheduler is not None:
            scheduler.step(val_loss)  # Step the scheduler with validation loss
    
    print("Training finished.")
    
    return model, train_losses, val_losses, val_accuracies

def evaluate_model(model, test_loader, num_rounds):
    """
    Evaluate the model on test data with detailed metrics
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader for test data
        num_rounds: Number of rounds in the data
        
    Returns:
        metrics: Dictionary containing all evaluation metrics
    """
    model.eval()
    predictions = []
    true_labels = []
    probabilities = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            # Forward pass
            output, _ = model(batch_x, num_rounds)
            
            # Get predictions and probabilities
            probs = output.squeeze(1)
            predicted = (probs > 0.5).float()
            
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(batch_y.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    probabilities = np.array(probabilities)
    
    # Calculate metrics
    accuracy = (predictions == true_labels).mean()
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    conf_matrix = confusion_matrix(true_labels, predictions)
    prob_min, prob_max = probabilities.min(), probabilities.max()
    
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print(f'Confusion Matrix:')
    print(conf_matrix)
    print(f'Probability range: [{prob_min:.3f}, {prob_max:.3f}]')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'predictions': predictions,
        'probabilities': probabilities,
        'prob_range': (prob_min, prob_max)
    }

def print_data_statistics(X_train, X_val, X_test, y_train, y_val, y_test):
    """Print dataset statistics"""
    total_samples = len(X_train) + len(X_val) + len(X_test)
    
    train_pct = len(X_train) / total_samples * 100
    val_pct = len(X_val) / total_samples * 100
    test_pct = len(X_test) / total_samples * 100
    
    train_balance = y_train.mean().item()
    val_balance = y_val.mean().item()
    test_balance = y_test.mean().item()
    
    print(f"Train: {len(X_train)} samples ({train_pct:.1f}%)")
    print(f"Val: {len(X_val)} samples ({val_pct:.1f}%)")
    print(f"Test: {len(X_test)} samples ({test_pct:.1f}%)")
    print(f"Train class balance: {train_balance:.3f}")
    print(f"Val class balance: {val_balance:.3f}")
    print(f"Test class balance: {test_balance:.3f}")

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
    if rounds == 5:
        path_detection = r"C:\Users\conof\Desktop\RNN-Decoder-QEC\google_qec3v5_experiment_data\surface_code_bX_d3_r05_center_3_5\detection_events.b8"
        path_obs = r"C:\Users\conof\Desktop\RNN-Decoder-QEC\google_qec3v5_experiment_data\surface_code_bX_d3_r05_center_3_5\obs_flips_actual.01"

    if rounds == 11:
        path_detection = r"C:\Users\conof\Desktop\RNN-Decoder-QEC\google_qec3v5_experiment_data\surface_code_bX_d3_r11_center_3_5\detection_events.b8"
        path_obs = r"C:\Users\conof\Desktop\RNN-Decoder-QEC\google_qec3v5_experiment_data\surface_code_bX_d3_r11_center_3_5\obs_flips_actual.01"

    if rounds == 17:
        path_detection = r"C:\Users\conof\Desktop\RNN-Decoder-QEC\google_qec3v5_experiment_data\surface_code_bX_d3_r17_center_3_5\detection_events.b8"
        path_obs = r"C:\Users\conof\Desktop\RNN-Decoder-QEC\google_qec3v5_experiment_data\surface_code_bX_d3_r17_center_3_5\obs_flips_actual.01"

    bits_per_shot_detection = rounds * 8

    with open(path_detection, "rb") as file:
        data_detection = file.read()

    with open(path_obs, "rb") as file:
        data_obs = file.read()

    detection = parse_b8(data_detection, bits_per_shot_detection)
    detection = torch.tensor(detection).reshape(50000, rounds, num_ancilla_qubits)

    obs = data_obs.replace(b"\n", b"")
    obs = list(obs)
    obs = [x - 48 for x in obs]

    return detection, obs

if __name__ == "__main__":
    # Configuration parameters
    distance = 3
    rounds = 5
    num_shots = 10000
    FineTune = False

    # Determine system size based on distance
    if distance == 3:
        num_qubits = 17
        num_data_qubits = 9
        num_ancilla_qubits = 8
    elif distance == 5:
        num_qubits = 49
        num_data_qubits = 25
        num_ancilla_qubits = 24

    # Load data from compressed file .npz
    detection_array1, observable_flips = load_data(num_shots)
    
    # Reorder using advanced indexing to create the chain connectivity
    order = [0,3,5,6,7,4,2,1]
    detection_array_ordered = detection_array1[..., order]

    # Load data from experimental .b8 file
    if FineTune:
        detection_array_exp, observable_flips_exp = load_data_exp()
        detection_array_ordered_exp = detection_array_exp[..., order]

    # Model hyperparameters
    input_size = 1
    hidden_size = 128
    output_size = 1
    chain_length = num_ancilla_qubits
    batch_size = 256
    test_size = 0.2
    val_size = 0.2
    learning_rate = 0.0005  # Changed to match your output
    patience = 2
    num_epochs = 30
    num_epochs_finetune = 1
    fc_layers_intra = [0]
    fc_layers_out = [int(hidden_size/8)]

    # Print configuration
    print(f"1D LSTM")
    print(f"Configuration: rounds={rounds}, distance={distance}, num_shots={num_shots}")
    print(f"Model parameters: hidden_size={hidden_size}, batch_size={batch_size}, fc_layers_intra = {fc_layers_intra}, fc_layers_out={fc_layers_out}")
    print(f"Training parameters: learning_rate={learning_rate}, num_epochs={num_epochs}")

    # Create data loaders with validation split
    train_loader, val_loader, test_loader, X_train, X_val, X_test, y_train, y_val, y_test = create_data_loaders(
        detection_array_ordered, observable_flips, batch_size, test_size, val_size)

    # Print data statistics
    print_data_statistics(X_train, X_val, X_test, y_train, y_val, y_test)

    # Create data loaders for experimental data if fine-tuning
    if FineTune:
        train_loader_exp, val_loader_exp, test_loader_exp, X_train_exp, X_val_exp, X_test_exp, y_train_exp, y_val_exp, y_test_exp = create_data_loaders(
            detection_array_ordered_exp, observable_flips_exp, batch_size, test_size, val_size)

    # Create model
    model = BlockRNN(input_size, hidden_size, output_size, chain_length, fc_layers_intra, fc_layers_out, batch_size)

    # Count and print model parameters
    num_params = count_parameters(model)
    print(f"Model has {num_params} parameters")

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience, verbose=False)

    # Train model
    start_time = time.time()
    model, train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, rounds, scheduler)
    end_time = time.time()

    # Fine-tune if requested
    if FineTune:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate/10)
        model, train_losses_ft, val_losses_ft, val_accuracies_ft = train_model(
            model, train_loader_exp, val_loader_exp, criterion, optimizer, num_epochs_finetune, rounds)

    # Print training time
    training_time_minutes = (end_time - start_time) / 60
    print(f"Training time: {training_time_minutes:.2f} minutes")

    # Evaluate model
    if FineTune:
        metrics = evaluate_model(model, test_loader_exp, rounds)
    else:
        metrics = evaluate_model(model, test_loader, rounds)

    # Save model (uncommented)
    # torch.save(model.state_dict(), "2D_LSTM_r17.pth")
    # print(f"Model saved to 2D_LSTM_r17.pth")