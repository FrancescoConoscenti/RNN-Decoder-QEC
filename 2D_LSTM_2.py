import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
import math
import time
from typing import List

class ImprovedLatticeRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.2):
        super(ImprovedLatticeRNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        # Use GRU instead of LSTM for simpler dynamics
        self.gru_cell = nn.GRUCell(input_size, hidden_size)
        
        # Attention mechanism for combining spatial information
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=4, 
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Process spatial neighbors
        self.neighbor_processor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # left + up
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Gate for controlling spatial influence
        self.spatial_gate = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),  # prev + spatial + input_proj
            nn.Sigmoid()
        )
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout_rate)
        self._initialize_weights()
        
    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                if 'gru' in name:
                    nn.init.orthogonal_(param.data)
                else:
                    nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
    
    def forward(self, x, hidden_states):
        device = x.device
        batch_size = x.size(0)
        
        hidden_left, hidden_up, hidden_prev = hidden_states
        
        # Initialize missing states
        if hidden_left is None:
            hidden_left = torch.zeros(batch_size, self.hidden_size, device=device)
        if hidden_up is None:
            hidden_up = torch.zeros(batch_size, self.hidden_size, device=device)
            
        # Project input
        x_proj = self.input_proj(x)
        
        # Process spatial neighbors
        if hidden_left is not None and hidden_up is not None:
            spatial_input = torch.stack([hidden_left, hidden_up], dim=1)  # [batch, 2, hidden]
            spatial_processed, _ = self.spatial_attention(spatial_input, spatial_input, spatial_input)
            spatial_combined = spatial_processed.mean(dim=1)  # Average attention output
        else:
            spatial_combined = torch.zeros(batch_size, self.hidden_size, device=device)
        
        # Compute spatial gate
        gate_input = torch.cat([hidden_prev, spatial_combined, x_proj], dim=1)
        spatial_gate = self.spatial_gate(gate_input)
        
        # Apply spatial influence
        enhanced_prev = hidden_prev + spatial_gate * spatial_combined
        
        # Update with GRU
        new_hidden = self.gru_cell(x, enhanced_prev)
        new_hidden = self.dropout(new_hidden)
        
        return new_hidden

class EnhancedLatticeRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, grid_height, grid_width, 
                 dropout_rate=0.3):
        super(EnhancedLatticeRNN, self).__init__()
        self.hidden_size = hidden_size
        self.grid_height = grid_height
        self.grid_width = grid_width
        
        # Create grid of cells
        self.cells = nn.ModuleList([
            ImprovedLatticeRNNCell(input_size, hidden_size, dropout_rate) 
            for _ in range(grid_height * grid_width)
        ])
        
        # Global context aggregation
        self.global_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Output processing with residual connection
        self.output_processor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, hidden_size),
)
        
    def forward(self, x, h_ext, grid_states):
        batch_size = x.size(0)
        device = x.device
        
        all_hidden_states = []
        
        # Process grid
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                cell_input = x[:, i, j].unsqueeze(1)
                h_prev = grid_states[i][j]
                
                # Get neighbors
                h_left = grid_states[i][j-1] if j > 0 else (h_ext if i == 0 and j == 0 else None)
                h_up = grid_states[i-1][j] if i > 0 else None
                
                # Process cell
                cell_index = i * self.grid_width + j
                h_new = self.cells[cell_index](cell_input, (h_left, h_up, h_prev))
                
                grid_states[i][j] = h_new
                all_hidden_states.append(h_new)
        
        # Global attention over all cell states
        hidden_stack = torch.stack(all_hidden_states, dim=1)  # [batch, num_cells, hidden]
        global_context, _ = self.global_attention(hidden_stack, hidden_stack, hidden_stack)
        
        # Use mean of global context for final prediction
        final_representation = global_context.mean(dim=1)
        
        # Generate output
        output = self.output_processor(final_representation)
        
        return output, grid_states[-1][-1], grid_states

class AdvancedBlockRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, grid_height, grid_width, 
                 dropout_rate=0.3):
        super(AdvancedBlockRNN, self).__init__()
        self.hidden_size = hidden_size
        self.grid_height = grid_height
        self.grid_width = grid_width
        
        # Temporal LSTM for processing sequences
        self.temporal_lstm = nn.LSTM(
            input_size=grid_height * grid_width,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Spatial processor
        self.spatial_rnn = EnhancedLatticeRNN(
            input_size, hidden_size, output_size, 
            grid_height, grid_width, dropout_rate
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x, num_rounds):
        batch_size = x.size(0)
        device = x.device
        
        # Temporal processing
        x_flat = x.view(batch_size, num_rounds, -1)  # Flatten spatial dimensions
        temporal_out, _ = self.temporal_lstm(x_flat)
        temporal_final = temporal_out[:, -1, :]  # Last timestep
        
        # Spatial processing on final timestep
        h_ext = torch.zeros(batch_size, self.hidden_size, device=device)
        grid_states = [[h_ext.clone() for _ in range(self.grid_width)] 
                      for _ in range(self.grid_height)]
        
        final_spatial_input = x[:, -1, :, :]  # Last round
        spatial_out, _, _ = self.spatial_rnn(final_spatial_input, h_ext, grid_states)
        
        # Fusion
        combined = torch.cat([temporal_final, spatial_out], dim=1)
        output = self.fusion(combined)
        
        return output, temporal_final

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()
    
def simple_create_data_loaders(detection_array, observable_flips, batch_size, test_size=0.2, val_size=0.2):
    """
    Simple version that works with your existing code structure
    Creates train/val/test splits
    
    Args:
        detection_array: Input data
        observable_flips: Target labels
        batch_size: Batch size for data loaders
        test_size: Fraction for test set (default 0.2 = 20%)
        val_size: Fraction for validation set from remaining data (default 0.2 = 20%)
    
    Returns:
        train_loader, val_loader, test_loader, X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Convert to numpy arrays if they're lists
    if isinstance(observable_flips, list):
        observable_flips = np.array(observable_flips)
    if isinstance(detection_array, list):
        detection_array = np.array(detection_array)
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        detection_array, observable_flips, 
        test_size=test_size, shuffle=True, random_state=42, 
        stratify=observable_flips
    )
    
    # Second split: separate train and validation from remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size, shuffle=True, random_state=42,
        stratify=y_temp
    )
    
    print(f"Data splits:")
    print(f"Train: {len(X_train)} samples ({len(X_train)/len(detection_array)*100:.1f}%)")
    print(f"Val: {len(X_val)} samples ({len(X_val)/len(detection_array)*100:.1f}%)")
    print(f"Test: {len(X_test)} samples ({len(X_test)/len(detection_array)*100:.1f}%)")
    
    # Check class balance in each split
    print(f"Train class balance: {np.mean(y_train):.3f}")
    print(f"Val class balance: {np.mean(y_val):.3f}")
    print(f"Test class balance: {np.mean(y_test):.3f}")
    
    # Create balanced data loaders
    train_loader = create_balanced_data_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = create_balanced_data_loader(X_val, y_val, batch_size, shuffle=False)
    test_loader = create_balanced_data_loader(X_test, y_test, batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, X_train, X_val, X_test, y_train, y_val, y_test


def create_balanced_data_loader(X, y, batch_size, shuffle=True):
    """Create balanced data loader using weighted sampling"""
    # Calculate class weights
    class_counts = np.bincount(y.astype(int))
    total_samples = len(y)
    
    # Inverse frequency weighting
    class_weights = total_samples / (2.0 * class_counts)
    sample_weights = class_weights[y.astype(int)]
    
    # Convert to tensors
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.tensor(y).float()
    
    dataset = TensorDataset(X_tensor, y_tensor)
    
    if shuffle:
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

def train_model(model, train_loader, val_loader, lr, num_epochs, num_rounds, patience, device='cuda',):
    model.to(device)
    
    # Use focal loss for imbalanced data
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Use AdamW with weight decay
    optimizer = optim.AdamW(model.parameters(), lr = lr, weight_decay=1e-4)
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, mode='min', factor=0.5, patience = patience, verbose=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output, _ = model(batch_x, num_rounds)
            loss = criterion(output.squeeze(), batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output, _ = model(batch_x, num_rounds)
                loss = criterion(output.squeeze(), batch_y)
                val_loss += loss.item()
                
                predicted = (torch.sigmoid(output.squeeze()) > 0.5).float()
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_accuracy = correct / total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']}")
        
        """
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        """
        
        scheduler.step(val_loss)
    
    # Load best model
    #model.load_state_dict(torch.load('best_model.pth'))
    return model, train_losses, val_losses

def evaluation(model, test_loader, num_rounds, device='cuda'):
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output, _ = model(batch_x, num_rounds)
            
            probs = torch.sigmoid(output.squeeze())
            predicted = (probs > 0.5).float()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = (all_predictions == all_targets).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='binary'
    )
    
    cm = confusion_matrix(all_targets, all_predictions)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Probability range: [{all_probs.min():.3f}, {all_probs.max():.3f}]")
    
    return accuracy, precision, recall, f1

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

def load_data_exp(rounds, num_ancilla_qubits):

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


# Usage example
def main():

    distance = 3
    rounds = 17
    num_shots = 3000

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load your data here
    detection_array1, observable_flips = load_data(num_shots, rounds)
    detection_array1_exp, observable_flips_exp = load_data_exp(rounds,  num_ancilla_qubits = 8)

    order = [0,5,1,3,4,6,2,7] # Reorder using advanced indexing to create the chain connectivity
    detection_array_ordered = detection_array1[..., order]
    observable_flips = observable_flips.astype(int).flatten().tolist()
    detection_array_ordered_exp = detection_array1_exp[..., order]
    observable_flips_exp = observable_flips_exp.astype(int).flatten().tolist()
    
    # Model hyperparameters - adjusted
    input_size = 1 
    hidden_size = 128  # Reduced size
    output_size = 1
    grid_height = 4
    grid_width = 2
    batch_size = 256  # Reduced batch size
    test_size = 0.2
    learning_rate = 0.001  # Increased learning rate
    learning_rate_fine = 0.0001  # Fine-tuning learning rate
    patience = 3  # Early stopping patience
    num_epochs = 2
    num_epochs_fine = 2  # Reduced fine-tuning epochs
    fc_layers_out = [hidden_size//2]  # Smaller output layers
    dropout_rate = 0.2
    
    print(f"2D LSTM")
    print(f"Configuration: rounds={rounds}, distance={distance}, num_shots={num_shots}")
    print(f"Model parameters: hidden_size={hidden_size}, batch_size={batch_size}")
    print(f"Training parameters: learning_rate={learning_rate}, num_epochs={num_epochs}")
    

    #adapt input data topology of model
    detection_array_2D = detection_array_ordered.reshape(num_shots, rounds, grid_height, grid_width)
    detection_array_2D_exp = detection_array_ordered_exp.reshape(50000, rounds, grid_height, grid_width)

    # Create data loaders
    train_loader, val_loader, test_loader, X_train, X_val, X_test, y_train, y_val, y_test = simple_create_data_loaders(
                                                        detection_array_2D, observable_flips, batch_size, test_size)
    
    train_loader_exp, val_loader_exp, test_loader_exp, X_train, X_val, X_test, y_train, y_val, y_test = simple_create_data_loaders(
                                                        detection_array_2D_exp, observable_flips_exp, batch_size, test_size)

    # Create model with improvements
    # Create model
    model = AdvancedBlockRNN(input_size, hidden_size, output_size, 
                            grid_height, grid_width, dropout_rate).to(device)
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    start_time = time.time()
    # Train model
    #model, losses = train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs, rounds, device)
    model, train_losses, val_losses = train_model(model, train_loader, val_loader, learning_rate, 
                                                            num_epochs, rounds, patience, device=device)
    end_time = time.time()
    print(f"Training time: {(end_time - start_time)/60:.2f} minutes")
    # Evaluate model
    accuracy, predictions, recall, f1 = evaluation(model, test_loader, rounds, device)

    #Finetune
    model, train_losses, val_losses = train_model(model, train_loader_exp, val_loader_exp, learning_rate_fine, 
                                                            num_epochs_fine, rounds, patience, device=device)
    
    accuracy, predictions, recall, f1 = evaluation(model, test_loader_exp, rounds, device)



if __name__ == "__main__":
    main()