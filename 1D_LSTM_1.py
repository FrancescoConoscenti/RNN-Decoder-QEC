import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import time
import sys
import math

class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, layers_sizes, hidden_size, dropout_rate=0.1):
        super(FullyConnectedNN, self).__init__()
        
        layers = []

        if layers_sizes == [0]:
            layers.append(nn.Linear(input_size, hidden_size))
            self._init_weights(layers[-1])
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        else:
            # Input layer
            layers.append(nn.Linear(input_size, layers_sizes[0]))
            self._init_weights(layers[-1])
            layers.append(nn.LayerNorm(layers_sizes[0]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
            # Hidden layers
            for i in range(len(layers_sizes) - 1):
                layers.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1]))
                self._init_weights(layers[-1])
                layers.append(nn.LayerNorm(layers_sizes[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
            
            # Output layer
            layers.append(nn.Linear(layers_sizes[-1], hidden_size))
            self._init_weights(layers[-1])

        # Combined sequential model
        self.network = nn.Sequential(*layers)
        
        # Skip connection for residual learning (when input and output sizes match)
        self.use_skip = (input_size == hidden_size)
        if not self.use_skip and len(layers_sizes) > 1:
            # Projection layer for skip connection when sizes don't match
            self.skip_projection = nn.Linear(input_size, hidden_size)
            self._init_weights(self.skip_projection)
            self.use_skip = True

    def _init_weights(self, layer):
        """Xavier/Glorot initialization for better gradient flow"""
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        output = self.network(x)
        
        # Apply skip connection if possible
        if self.use_skip:
            if hasattr(self, 'skip_projection'):
                skip = self.skip_projection(x)
            else:
                skip = x
            output = output + skip
            
        return output


class LatticeRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, fc_layers, batch_size, dropout_rate=0.1):
        """
        Custom RNN cell that processes inputs in a 2D lattice structure
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            fc_layers: List of layer sizes for fully connected networks
            batch_size: Batch size for training
            dropout_rate: Dropout probability
        """
        super(LatticeRNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        
        # Process input with layer normalization
        self.fc_input = nn.Linear(input_size, input_size)
        self._init_weights(self.fc_input)
        self.input_norm = nn.LayerNorm(input_size)
        
        # Process combined hidden states 
        self.hidden_processor = FullyConnectedNN(hidden_size*2, fc_layers, hidden_size, dropout_rate)
        self.cell_processor = FullyConnectedNN(hidden_size*2, fc_layers, hidden_size, dropout_rate)
        
        # LSTM cell for time dimension with layer normalization
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self._init_lstm_weights()
        
        # Layer normalization for LSTM outputs
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.cell_norm = nn.LayerNorm(hidden_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
    def _init_weights(self, layer):
        """Xavier initialization"""
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
                
    def _init_lstm_weights(self):
        """Initialize LSTM weights with orthogonal initialization for recurrent weights"""
        for name, param in self.lstm_cell.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                # Initialize forget gate bias to 1 for better gradient flow
                n = param.size(0)
                start, end = n//4, n//2
                param.data[start:end].fill_(1.)
        
    def forward(self, x, hidden_states):
        """
        Forward pass for the lattice RNN cell
        
        Args:
            x: Input tensor [batch_size, input_size]
            hidden_states: Tuple containing:
                - hidden: Hidden state from spatial neighbor
                - cell: Cell state from spatial neighbor  
                - hidden_prev: Previous hidden state in time
                - cell_prev: Previous cell state in time
                
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
        
        # Apply layer normalization
        processed_h = self.hidden_norm(processed_h)
        processed_c = self.cell_norm(processed_c)
        
        # Process input
        x = x.squeeze(1).float()
        x = self.input_norm(self.fc_input(x))
        
        # Update hidden state using LSTM cell
        hidden, cell = self.lstm_cell(x, (processed_h, processed_c))
        
        # Apply dropout during training
        if self.training:
            hidden = self.dropout(hidden)
            cell = self.dropout(cell)
        
        return hidden, cell

class LatticeRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, length, fc_layers_intra, fc_layers_out, batch_size, dropout_rate=0.1):
        """
        Network that processes inputs in a 2D lattice structure
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            output_size: Size of output
            length: Length of the chain
            fc_layers_intra: List of layer sizes for intra-cell fully connected networks
            fc_layers_out: List of layer sizes for output fully connected networks
            batch_size: Batch size for training
            dropout_rate: Dropout probability
        """
        super(LatticeRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.chain_length = length
        self.dropout_rate = dropout_rate
        
        # Create a chain of RNN cells
        self.cells = nn.ModuleList([
            LatticeRNNCell(input_size, hidden_size, fc_layers_intra, batch_size, dropout_rate) 
            for _ in range(self.chain_length)
        ])
        
        # Output processing with enhanced architecture
        self.fc_out = FullyConnectedNN(hidden_size*2, fc_layers_out, hidden_size, dropout_rate)
        
        # Final output layer with proper initialization
        self.final_output = nn.Linear(hidden_size, output_size)
        self._init_weights(self.final_output)
        
        # Layer normalization and activation
        self.output_norm = nn.LayerNorm(hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)
        
    def _init_weights(self, layer):
        """Xavier initialization"""
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x, h_ext, c_ext, chain_states):
        """
        Forward pass for the lattice RNN
        
        Args:
            x: Input tensor [batch_size, chain_length, 1]
            h_ext: External hidden state
            c_ext: External cell state
            chain_states: Previous states for the chain
            
        Returns:
            output: Output tensor
            final_h: Final hidden state
            final_c: Final cell state
            chain_states: Updated chain states
        """
        batch_size = x.size(0)
        device = x.device
        x = x.squeeze(2)
        
        # Process each cell in the chain
        for i in range(self.chain_length):
            
            # Get input for current cell
            cell_input = x[:, i].unsqueeze(1).unsqueeze(1)
                
            # Get temporal states from previous round
            h_time, c_time = chain_states[i]
                
            # Handle spatial connections
            if i == 0:
                h_space = h_ext
                c_space = c_ext   
            else:
                h_space, c_space = chain_states[i-1]
            
            # Process through cell
            h_new, c_new = self.cells[i](cell_input, (h_space, c_space, h_time, c_time))
                
            # Update chain state
            chain_states[i] = (h_new, c_new)
        
        # Get final states
        final_h, final_c = chain_states[-1]
        
        # Combine final states
        final = torch.cat((final_h, final_c), dim=1)
        
        # Generate output with proper normalization and dropout
        output = self.fc_out(final)
        output = self.output_norm(output)
        output = self.dropout(output)
        output = self.final_output(output)
        output = self.sigmoid(output)
        
        return output, final_h, final_c, chain_states

class BlockRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, chain_length, fc_layers_intra, fc_layers_out, batch_size, dropout_rate=0.1):
        """
        Block RNN model that processes multiple time steps of data on a 2D lattice
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            output_size: Size of output
            chain_length: Length of the processing chain
            fc_layers_intra: List of layer sizes for intra-cell networks
            fc_layers_out: List of layer sizes for output networks
            batch_size: Batch size for training
            dropout_rate: Dropout probability
        """
        super(BlockRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.chain_length = chain_length
        self.dropout_rate = dropout_rate
        
        # Input processing with proper initialization
        self.fc_in = nn.Linear(input_size, input_size)
        self._init_weights(self.fc_in)
        self.input_norm = nn.LayerNorm(input_size)
        
        # Lattice RNN for spatial processing
        self.rnn_block = LatticeRNN(input_size, hidden_size, output_size, chain_length, 
                                   fc_layers_intra, fc_layers_out, batch_size, dropout_rate)
        
        # Gradient clipping value
        self.max_grad_norm = 1.0
        
    def _init_weights(self, layer):
        """Xavier initialization"""
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x, num_rounds):
        """
        Forward pass for the Block RNN
        
        Args:
            x: Input tensor [batch_size, num_rounds, chain_length]
            num_rounds: Number of time steps to process
            
        Returns:
            output: Final prediction
            final_h: Final hidden state
        """
        batch_size = x.size(0)
        device = x.device
        
        # Initialize external hidden states with proper scaling
        init_std = 1.0 / math.sqrt(self.hidden_size)
        h_ext = torch.randn(batch_size, self.hidden_size, device=device) * init_std
        c_ext = torch.randn(batch_size, self.hidden_size, device=device) * init_std
        
        # Initialize chain states
        chain_states = [(h_ext.clone(), c_ext.clone()) for _ in range(self.chain_length)]
        
        # Process each round
        for round_idx in range(num_rounds):
            # Get input for this round
            round_input = x[:, round_idx, :].unsqueeze(2)
            
            # Process input
            round_input = round_input.squeeze(2)  # [batch_size, chain_length]
            round_input = round_input.unsqueeze(-1)  # [batch_size, chain_length, 1]
            round_input = round_input.reshape(-1, 1)  # [batch_size * chain_length, 1]
            round_input = self.fc_in(round_input)     # [batch_size * chain_length, 1]
            round_input = self.input_norm(round_input)  # [batch_size * chain_length, 1]
            round_input = round_input.reshape(batch_size, self.chain_length, 1)  # [batch_size, chain_length, 1]
            
            # Process through lattice RNN
            output, h_ext, c_ext, chain_states = self.rnn_block(round_input, h_ext, c_ext, chain_states)
        
        return output, h_ext
    
    def clip_gradients(self):
        """Clip gradients to prevent exploding gradients"""
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)

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
     
    if isinstance(detection_array, (np.ndarray, list)):
        # Convert to PyTorch tensors
        X_train = torch.from_numpy(X_train).float()
        X_test = torch.from_numpy(X_test).float()

    if isinstance(observable_flips, (np.ndarray, list)):
        y_train = torch.tensor(y_train).float()
        y_test = torch.tensor(y_test).float()
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, drop_last=False  # Changed shuffle to True for better training
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, 
        shuffle=False, drop_last=False
    )
    
    return train_loader, test_loader, X_train, X_test, y_train, y_test

def train_model(model, train_loader, criterion, optimizer, patience, num_epochs, num_rounds, scheduler=None):
    """
    Train the model with enhanced training loop
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train for
        num_rounds: Number of rounds in the data
        scheduler: Learning rate scheduler (optional)
        
    Returns:
        model: Trained model
        losses: List of losses per epoch
    """
    model.train()
    losses = []
    best_loss = float('inf')
    patience = patience
    patience_counter = 0
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output, _ = model(batch_x, num_rounds)
            loss = criterion(output.squeeze(1), batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            model.clip_gradients()
            
            # Check for NaN gradients
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"Warning: NaN gradient detected in {name}")
                    param.grad.zero_()  # Zero out the NaN gradients
            
            # Optimizer step
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
        
        # Calculate average loss for this epoch
        avg_loss = running_loss / num_batches
        losses.append(avg_loss)
        
        # Learning rate scheduling
        scheduler.step(running_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        
        print(f"Epoch [{epoch+1}/{num_epochs}], LR: {current_lr}, Loss: {avg_loss:.4f}")
        
        # Early stopping
        if patience_counter >= patience and epoch > 10:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print("Training finished.")
    return model, losses

def evaluate_model(model, test_loader, num_rounds):
    """
    Evaluate the model on test data
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader for test data
        num_rounds: Number of rounds in the data
        
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

def parse_b8(data: bytes, bits_per_shot: int):
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
    rounds = 11
    num_shots = 100000
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
    learning_rate = 0.002
    patience = 2
    num_epochs = 20  # Increased for early stopping
    num_epochs_finetune = 5
    fc_layers_intra = [0]
    fc_layers_out = [int(hidden_size/4)]  # Slightly larger for better capacity
    dropout_rate = 0.1

    # Print configuration
    print(f"1D LSTM_1")
    print(f"Configuration: rounds={rounds}, distance={distance}, num_shots={num_shots}")
    print(f"Model parameters: hidden_size={hidden_size}, batch_size={batch_size}")
    print(f"FC layers: intra={fc_layers_intra}, out={fc_layers_out}, dropout={dropout_rate}")
    print(f"Training parameters: learning_rate={learning_rate}, num_epochs={num_epochs}")

    # Create data loaders
    train_loader, test_loader, X_train, X_test, y_train, y_test = create_data_loaders(
        detection_array_ordered, observable_flips, batch_size, test_size)

    # Create data loaders for fine-tuning
    if FineTune:
        train_loader_exp, test_loader_exp, X_train_exp, X_test_exp, y_train_exp, y_test_exp = create_data_loaders(
            detection_array_ordered_exp, observable_flips_exp, batch_size, test_size)

    # Create model with enhanced features
    model = BlockRNN(input_size, hidden_size, output_size, chain_length, 
                     fc_layers_intra, fc_layers_out, batch_size, dropout_rate)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Define loss function and optimizer with weight decay
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience, verbose=True
    )

    # Train model
    start_time = time.time()
    model, losses = train_model(model, train_loader, criterion, optimizer, patience,
                               num_epochs, rounds, scheduler)
    end_time = time.time()

    # Fine-tune if specified
    if FineTune:
        print("\nStarting fine-tuning on experimental data...")
        optimizer_ft = optim.AdamW(model.parameters(), lr=learning_rate/10, weight_decay=1e-5)
        scheduler_ft = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_ft, mode='min', factor=0.7, patience=patience, verbose=True)

        model, losses_ft = train_model(model, train_loader_exp, criterion, optimizer_ft, 
                                      num_epochs_finetune, rounds, scheduler_ft)

    # Evaluate model
    print("\nEvaluating on test data...")
    accuracy, predictions = evaluate_model(model, test_loader, rounds)
    
    if FineTune:
        print("Evaluating on experimental test data...")
        accuracy_exp, predictions_exp = evaluate_model(model, test_loader_exp, rounds)

    # Print execution time
    print(f"\nExecution time: {end_time - start_time:.2f} seconds")

    # Save model with additional metadata
    model_save_path = f"enhanced_model_r{rounds}_d{distance}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size,
            'chain_length': chain_length,
            'fc_layers_intra': fc_layers_intra,
            'fc_layers_out': fc_layers_out,
            'dropout_rate': dropout_rate
        },
        'training_config': {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'accuracy': accuracy
        },
        'losses': losses
    }, model_save_path)
    print(f"Enhanced model saved to {model_save_path}")