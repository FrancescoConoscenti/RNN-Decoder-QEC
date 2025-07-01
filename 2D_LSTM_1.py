import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout_rate=0.1):
        super(FullyConnectedNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for size in hidden_layers:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.BatchNorm1d(size))  # Add batch normalization
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))  # Add dropout
            prev_size = size
            
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class LatticeRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, dropout_rate=0.1):
        super(LatticeRNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        
        # Process input
        self.fc_input = nn.Linear(input_size, input_size)
        
        # Process combined hidden states with proper initialization
        self.hidden_processor = nn.Linear(hidden_size*3, hidden_size)
        self.cell_processor = nn.Linear(hidden_size*3, hidden_size)
        
        # Add layer normalization
        self.layer_norm_h = nn.LayerNorm(hidden_size)
        self.layer_norm_c = nn.LayerNorm(hidden_size)
        
        # LSTM cell for time dimension
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        
        # Add dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights properly
        self._initialize_weights()
        
    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() >= 2:  # Only apply to 2D+ tensors
                    if 'lstm' in name:
                        # Initialize LSTM weights with Xavier/Glorot initialization
                        nn.init.xavier_uniform_(param.data)
                    else:
                        nn.init.kaiming_normal_(param.data, nonlinearity='relu')
                else:
                    # For 1D weight tensors (rare cases)
                    nn.init.normal_(param.data, 0, 0.01)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
    
    def forward(self, x, hidden_states):
        device = x.device
        batch_size = x.size(0)  # Use actual batch size
        
        hidden_left, cell_left, hidden_up, cell_up, hidden_prev, cell_prev = hidden_states
        
        # Initialize missing hidden states with proper batch size
        if hidden_left is None:
            hidden_left = torch.zeros(batch_size, self.hidden_size, device=device)
            cell_left = torch.zeros(batch_size, self.hidden_size, device=device)
        if hidden_up is None:
            hidden_up = torch.zeros(batch_size, self.hidden_size, device=device)
            cell_up = torch.zeros(batch_size, self.hidden_size, device=device)
            
        # Combine hidden states from different directions
        combined_h = torch.cat((hidden_left, hidden_up, hidden_prev), dim=1)
        combined_c = torch.cat((cell_left, cell_up, cell_prev), dim=1)
        
        # Process combined hidden states with normalization
        processed_h = self.layer_norm_h(self.hidden_processor(combined_h))
        processed_c = self.layer_norm_c(self.cell_processor(combined_c))
        
        # Apply dropout
        processed_h = self.dropout(processed_h)
        processed_c = self.dropout(processed_c)
        
        # Update hidden state using LSTM cell
        if x.dim() > 2:
            x = x.squeeze(1).float()
        
        hidden, cell = self.lstm_cell(x, (processed_h, processed_c))
        
        return hidden, cell

class LatticeRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, grid_height, grid_width, 
                 fc_layers_out, batch_size, dropout_rate=0.1):
        super(LatticeRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.bn =  nn.BatchNorm1d(output_size)# Batch normalization for output
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for output

        # Create a grid of RNN cells
        self.cells = nn.ModuleList([
            LatticeRNNCell(input_size, hidden_size, batch_size, dropout_rate) 
            for _ in range(grid_height * grid_width)
        ])
        
        # Output layer with improved architecture
        if fc_layers_out and fc_layers_out[0] > 0:
            self.fc_out = FullyConnectedNN(hidden_size*2, fc_layers_out, output_size, dropout_rate)
        else:
            self.fc_out = nn.Linear(hidden_size, output_size)
        
        # Remove sigmoid from here - will be applied in loss function
        
    def forward(self, x, h_ext, c_ext, grid_states):
        batch_size = x.size(0)
        device = x.device
       
        # Process each cell in the grid
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                # Get input for current cell
                cell_input = x[:, i, j].unsqueeze(1)
                
                # Get previous states for this cell
                h_prev, c_prev = grid_states[i][j]
                
                # Handle boundary conditions properly
                if i == 0 and j == 0:
                    h_left = h_ext
                    c_left = c_ext
                elif j > 0:
                    h_left, c_left = grid_states[i][j-1]
                else:
                    h_left, c_left = None, None
                
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

        final_h = final_h + h_ext   
        final_c = final_c + c_ext
        
        final = torch.cat((final_h, final_c), dim=1)
        
        # Generate output
        output = self.fc_out(final)
        output = self.bn(output)
        output = self.sigmoid(output)
        
        return output, final_h, final_c, grid_states

class BlockRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, grid_height, grid_width, 
                 fc_layers_out, batch_size, dropout_rate=0.1):
        super(BlockRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.grid_height = grid_height
        self.grid_width = grid_width
        
        # Input processing
        self.fc_in = nn.Linear(input_size, input_size)
        
        # Lattice RNN for spatial processing
        self.rnn_block = LatticeRNN(
            input_size, hidden_size, output_size, 
            grid_height, grid_width, fc_layers_out, batch_size, dropout_rate
        )
    
    def forward(self, x, num_rounds):
        batch_size = x.size(0)
        device = x.device
        
        # Initialize external hidden states
        h_ext = torch.zeros(batch_size, self.hidden_size, device=device)
        c_ext = torch.zeros(batch_size, self.hidden_size, device=device)
        
        # Initialize grid states properly
        grid_states = [
            [(h_ext.clone(), c_ext.clone()) for _ in range(self.grid_width)] 
            for _ in range(self.grid_height)
        ]
        
        # Process each round
        for round_idx in range(num_rounds):
            round_input = x[:, round_idx, :, :]
            
            # Process through lattice RNN
            output, h_ext, c_ext, grid_states = self.rnn_block(
                round_input, h_ext, c_ext, grid_states
            )
        
        return output, h_ext

def check_data_balance(y_data):
    """Check class balance in the data"""
    unique, counts = np.unique(y_data, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")
    print(f"Class balance: {counts[1]/(counts[0]+counts[1]):.3f}")
    return counts[1]/(counts[0]+counts[1])

def create_data_loaders(detection_array, observable_flips, batch_size, test_size=0.2):
    # Check data balance
    balance = check_data_balance(observable_flips)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        detection_array, observable_flips, 
        test_size=test_size, shuffle=True, random_state=42  # Enable shuffling
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
        shuffle=True, drop_last=True  # Enable shuffling
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, 
        shuffle=False, drop_last=True
    )
    
    return train_loader, test_loader, X_train, X_test, y_train, y_test, balance

def train_model(model, train_loader, criterion, optimizer, scheduler, 
                              num_epochs, num_rounds, device='cuda'):
    model.to(device)
    model.train()
    losses = []
    
    # Add gradient clipping
    max_grad_norm = 1.0
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_samples = 0
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output, _ = model(batch_x, num_rounds)
            output = output.squeeze()
            
            # Calculate loss
            loss = criterion(output, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            running_loss += loss.item()
            total_samples += batch_y.size(0)
            
            # Debug: Print statistics every 100 batches
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}, "
                      f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Calculate average loss for this epoch
        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)
        
        # Step scheduler
        if scheduler:
            scheduler.step(avg_loss)
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
        else:
            current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{num_epochs}], lr: {current_lr:.6f}, Loss: {avg_loss:.4f}")
        
        # Early stopping check
        if len(losses) > 5:
            recent_losses = losses[-5:]
            if max(recent_losses) - min(recent_losses) < 1e-6:
                print("Loss plateau detected, consider adjusting learning rate or model architecture")
    
    print("Training finished.")
    return model, losses

def evaluate_model(model, test_loader, num_rounds, device='cuda'):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    predictions = []
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            output, _ = model(batch_x, num_rounds)
            output = output.squeeze()
            
            # Apply sigmoid for probability
            prob_output = torch.sigmoid(output)
            
            # Get predictions
            predicted = (prob_output > 0.5).float()
            predictions.extend(predicted.cpu().numpy())
            all_outputs.extend(prob_output.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            
            # Calculate accuracy
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
    
    accuracy = correct / total
    
    # Additional statistics
    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets)
    
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    print(f'Output range: [{all_outputs.min():.3f}, {all_outputs.max():.3f}]')
    print(f'Mean output: {all_outputs.mean():.3f}')
    print(f'Std output: {all_outputs.std():.3f}')
    
    return accuracy, predictions, all_outputs

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


# Example usage modifications:
def main():
    # Your existing configuration
    distance = 3
    rounds = 11
    num_shots = 30000
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load your data here
    detection_array1, observable_flips = load_data(num_shots, rounds)

    order = [0,5,1,3,4,6,2,7] # Reorder using advanced indexing to create the chain connectivity
    detection_array_ordered = detection_array1[..., order]
    observable_flips = observable_flips.astype(int).flatten().tolist()
    
    
    # Model hyperparameters - adjusted
    input_size = 1
    hidden_size = 128  # Reduced size
    output_size = 1
    grid_height = 4
    grid_width = 2
    batch_size = 256  # Reduced batch size
    test_size = 0.2
    learning_rate = 0.001  # Increased learning rate
    num_epochs = 20
    fc_layers_out = [hidden_size//2]  # Smaller output layers
    dropout_rate = 0.2
    
    print(f"Configuration: rounds={rounds}, distance={distance}, num_shots={num_shots}")
    print(f"Model parameters: hidden_size={hidden_size}, batch_size={batch_size}")
    print(f"Training parameters: learning_rate={learning_rate}, num_epochs={num_epochs}")
    

    #adapt input data topology of model
    detection_array_2D = detection_array_ordered.reshape(num_shots, rounds, grid_height, grid_width)

    # Create data loaders
    train_loader, test_loader, X_train, X_test, y_train, y_test, balance = create_data_loaders(
    detection_array_2D, observable_flips, batch_size, test_size)

    # Create model with improvements
    model = BlockRNN(input_size, hidden_size, output_size, grid_height, 
                     grid_width, fc_layers_out, batch_size, dropout_rate).to(device)
    
    # Use BCEWithLogitsLoss instead of BCELoss (more numerically stable)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Train model
    model, losses = train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs, rounds, device)

    # Evaluate model
    accuracy, predictions, all_outputs = evaluate_model(model, test_loader, rounds, device)

    
if __name__ == "__main__":
    main()