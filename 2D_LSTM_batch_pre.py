import stim
import numpy as np
from sklearn.model_selection import train_test_split

distance=3
rounds=5

if distance ==3:
    num_qubits=17
    num_data_qubits=9
    num_ancilla_qubits=8

if distance ==5:
    num_qubits=49
    num_data_qubits=25
    num_ancilla_qubits=24

path = r"google_qec3v5_experiment_data/surface_code_bX_d3_r05_center_3_5/circuit_noisy.stim"
circuit_google = stim.Circuit.from_file(path)

circuit_surface = stim.Circuit.generated(
    "surface_code:rotated_memory_x",
    rounds=rounds,
    distance=distance,
    after_clifford_depolarization=0.01,
    after_reset_flip_probability=0.01,
    before_measure_flip_probability=0.01,
    before_round_data_depolarization=0.01)

##################################################################################################à

num_shots=2000000 #batch size multiple
"""# Compile the sampler
sampler = circuit_surface.compile_detector_sampler()
# Sample shots, with observables
detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)


detection_events = detection_events.astype(int)
detection_strings = [''.join(map(str, row)) for row in detection_events] #compress the detection events in a tensor
detection_events_numeric = [[int(value) for value in row] for row in detection_events] # Convert string elements to integers (or floats if needed)
detection_array = np.array(detection_events_numeric) # Convert detection_events to a numpy array

detection_array1 = detection_array.reshape(num_shots, rounds, num_ancilla_qubits) #first dim is the number of shots, second dim round number, third dim is the Ancilla 

observable_flips = observable_flips.astype(int).flatten().tolist()"""

loaded_data = np.load('data_stim/google_r5.npz')
detection_array1 = loaded_data['detection_array1']
observable_flips = loaded_data['observable_flips']

############################################################################################################

import torch
import torch.nn as nn
import torch.optim as optim


class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, layers_sizes, hidden_size):
        super(FullyConnectedNN, self).__init__()
        
        # Define the input layer
        self.input_layer = nn.Linear(input_size, layers_sizes[0])
        
        # Define hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(layers_sizes) - 1):
            self.hidden_layers.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1]))
        
        # Define output layer
        self.output_layer = nn.Linear(layers_sizes[-1], hidden_size)

        # Define activation function (e.g., ReLU)
        self.activation = nn.ReLU()

    def forward(self, x):
        # Pass through input layer
        #torch.transpose(x.squeeze(0), 1,0)

        x = self.activation(self.input_layer(x))
        
        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        
        # Pass through output layer
        x = self.output_layer(x)
        
        return x
    
class LatticeRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size,layers_sizes,batch_size):
        super(LatticeRNNCell, self).__init__()
        self.hidden=hidden_size
        self.batch_size=batch_size
        self.fc_input = nn.Linear(input_size, input_size)
        #self.fc_hidden_double = nn.Linear(hidden_size*3, hidden_size)
        self.multi_layer_fc = FullyConnectedNN(hidden_size*3, layers_sizes, hidden_size)
        self.multi_layer_fc1 = FullyConnectedNN(hidden_size*3, layers_sizes, hidden_size)
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)

    def forward(self, x, hidden_left, hidden_up, hidden_pre, hidden_pre1):
        # if hidden_left is not None and hidden_bottom is not None:
        #     hidden = (hidden_left + hidden_bottom) / 2  # Average of left and bottom
        # elif hidden_left is not None:
        #     hidden = hidden_left
        # elif hidden_bottom is not None:
        #     hidden = hidden_bottom
        # else:
        #     hidden = torch.zeros(x.size(0), self.hidden_size).to(x.device)  # Initial hidden state
        
        #input fc net
        #input=self.fc_input(x.type(torch.FloatTensor))

        """
        # Combine the hidden states from left and bottom
        if hidden_left is not None and hidden_up is not None:
            combined_hidden=torch.cat((hidden_left,hidden_up),1)
            hidden=self.fc_hidden_double(combined_hidden.squeeze(0)).unsqueeze(0)

        elif hidden_left is not None:
            hidden=self.fc_hidden_single(hidden_left.squeeze(0)).unsqueeze(0)

        elif hidden_up is not None:
            hidden=self.fc_hidden_single(hidden_up.squeeze(0)).unsqueeze(0)
        else:
            hidden=torch.zeros(1,self.hidden, dtype=torch.float)
        """
        
        hidden_pre=torch.Tensor(hidden_pre)
        hidden_pre1=torch.Tensor(hidden_pre1)

        # Combine the hidden states from left and bottom
        if hidden_left[0] is not None and hidden_up[0] is not None:
            combined_hidden=torch.cat((hidden_left[0],hidden_up[0], hidden_pre),1)
        elif hidden_left[0] is not None:
            hidden_init_up=torch.zeros(self.batch_size,self.hidden, dtype=torch.float)
            combined_hidden=torch.cat((hidden_left[0],hidden_init_up, hidden_pre),1)
        elif hidden_up[0] is not None:
            hidden_init_left=torch.zeros(self.batch_size,self.hidden, dtype=torch.float)
            combined_hidden=torch.cat((hidden_init_left,hidden_up[0], hidden_pre),1)
        else:
            hidden_init_left=torch.zeros(self.batch_size,self.hidden, dtype=torch.float)
            hidden_init_up=torch.zeros(self.batch_size,self.hidden, dtype=torch.float)
            combined_hidden=torch.cat((hidden_init_left,hidden_init_up, hidden_pre),1)



        if hidden_left[1] is not None and hidden_up[1] is not None:
            combined_hidden1=torch.cat((hidden_left[1],hidden_up[1], hidden_pre1),1)
        elif hidden_left[1] is not None:
            hidden_up=torch.zeros(self.batch_size,self.hidden, dtype=torch.float)
            combined_hidden1=torch.cat((hidden_left[1],hidden_up, hidden_pre1),1)
        elif hidden_up[1] is not None:
            hidden_left=torch.zeros(self.batch_size,self.hidden, dtype=torch.float)
            combined_hidden1=torch.cat((hidden_left,hidden_up[1], hidden_pre1),1)
        else:
            hidden_left=torch.zeros(self.batch_size,self.hidden, dtype=torch.float)
            hidden_up=torch.zeros(self.batch_size,self.hidden, dtype=torch.float)
            combined_hidden1=torch.cat((hidden_left,hidden_up, hidden_pre1),1)


        hidden=self.multi_layer_fc(combined_hidden)
        hidden1=self.multi_layer_fc1(combined_hidden1)

        x=x.squeeze(1).float()

        # Update hidden state using current input and combined hidden state
        hidden,hidden1 = self.lstm_cell(x, (hidden,hidden1))
        
        return hidden,hidden1

class LatticeRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, grid_height, grid_width,layers_sizes, batch_size):
        super(LatticeRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.grid_height = grid_height
        self.grid_width = grid_width 
        self.lstm_cells = nn.ModuleList([LatticeRNNCell(input_size, hidden_size, layers_sizes, batch_size) for _ in range(grid_height * grid_width)])
        self.fc_hidden = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # For binary output
    
    def forward(self, x, hidden_ext, hidden_ext1, grid):
        
        # Reshape the input to match the grid size
        batch_size, seq_len, _ = x.size()
        x = x.reshape(batch_size,self.grid_height, self.grid_width)

        for i in range(self.grid_height):
            for j in range(self.grid_width):

                input_bit = x[:,i,j].unsqueeze(1).unsqueeze(1)  # Get the input for the current cell
                (hidden_pre, hidden_pre1)=grid[i][j]

                if i==0 and j==0:
                    hidden_left=(hidden_ext,hidden_ext1)

                if j > 0 :
                    hidden_left = grid[i][j - 1]
                else:
                    hidden_left = (None,None)

                if i > 0:
                    hidden_up = grid[i-1][j]
                else:
                    hidden_up = (None,None)

                

                # Get the index for the current RNN cell
                cell_index = i * self.grid_width + j
                hidden, hidden1 = self.lstm_cells[cell_index](input_bit, hidden_left, hidden_up, hidden_pre, hidden_pre1)

                # Store the hidden state in the grid
                grid[i][j] = (hidden,hidden1)

        # The output that matters is the hidden state from the top-right corner (i.e., grid[grid_size-1][grid_size-1])
        final_hidden,  final_hidden1 = grid[-1][-1]

        # Pass the hidden state through the fully connected layer and sigmoid for binary output
        #hidden= self.fc_hidden(bottom_right_hidden)
        output = self.fc_out(final_hidden)
        output = self.sigmoid(output)

        return output,  final_hidden, final_hidden1, grid
    


# RNN model
class BlockRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, grid_height, grid_width, rounds,layers_sizes,batch_size):
        super(BlockRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.fc_in = nn.Linear(input_size, input_size)
        self.rnn_block = LatticeRNN(input_size, hidden_size, output_size, grid_height, grid_width, layers_sizes,batch_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # For binary output
    
    def forward(self, x, rounds):
        #initialize the external hidden states
        hidden_ext = torch.zeros(self.batch_size,self.hidden_size) # (1,hidden_size) hidden state of the previous round 
        hidden_ext1 = torch.zeros(self.batch_size,self.hidden_size)

        #Initialize the previous hidden state for each element of the grid
        grid = [[(hidden_ext,hidden_ext1) for _ in range(grid_width)] for _ in range(grid_height)]

        for round in range (rounds):
            input_block = x[:,round,:].unsqueeze(2)  # (1, 8) syndromes
            out, hidden_ext, hidden_ext1,grid = self.rnn_block(input_block, hidden_ext, hidden_ext1, grid)

        #I already use fc, sigmoid in the LatticeRNN
        #out = self.fc_out(out)  # Use the last time-step's output, needed for changing the dimension of the output compared of input
        #out = self.sigmoid(out)  # I need a Binary output
        return out, hidden_ext[0]
    
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)
    
###############################################################################################################

def train_rnn(model, X_train, y_train, criterion, optimizer, num_epochs, batch_size,rounds):
    
    model.train()  # Set the model to training mode
    # If batch_size is None, set it to the full size of the dataset (no mini-batching)

    num_batches = len(X_train[:,0,0]) // batch_size
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        # # Shuffle the dataset at the beginning of each epoch
        # permutation = torch.randperm(num_samples)
        # X_train = X_train[permutation]
        # y_train = y_train[permutation]

        # Mini-batch training loop
        for batch_idx in range(num_batches):
            # Create mini-batches
            batch_x = torch.from_numpy(X_train[batch_idx * batch_size : (batch_idx + 1) * batch_size])
            batch_y = torch.Tensor(y_train[batch_idx * batch_size : (batch_idx + 1) * batch_size])

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            #hidden = model.init_hidden(1)
            output, hidden= model(batch_x, rounds)

            # Compute loss
            loss = criterion(output.squeeze(1), batch_y)

            # Backward pass (compute gradients)
            loss.backward()

            # Optimize (update weights)
            optimizer.step()

            # Accumulate the loss for logging purposes
            running_loss += loss.item()

        # Print average loss after each epoch
        avg_loss = running_loss / num_batches
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    print("Training finished.")

def binary_array_to_tensor(binary_array):
    # Check if the input is a NumPy array, if not, convert it
    if isinstance(binary_array, np.ndarray):
        tensor = torch.from_numpy(binary_array).float()  # Convert NumPy array to float32 tensor
    else:
        # If not a NumPy array, convert it as before
        tensor = torch.tensor([[int(bit) for bit in binary_array]], dtype=torch.float32)
    return tensor.unsqueeze(0)  # Add batch dimension (batch_size = 1)


def test(model, test_sequences, targets,batch_size):
    model.eval()  # Set the model to evaluation mode (disable dropout, etc.)
    correct = 0

    hidden = None
    num_batches = len(test_sequences[:,0,0]) // batch_size
    
    
    with torch.no_grad():  # Disable gradient computation for testing
        for batch_idx in range(num_batches):
        
            output=np.zeros(batch_size)
            batch_x = torch.from_numpy(test_sequences[batch_idx * batch_size : (batch_idx + 1) * batch_size])
            target = targets[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            rounds = len(batch_x[0,:,0])
            
            # Initialize hidden state
            #if hidden is None:
                # Initialize hidden state for the first sequence (or batch)
            #    hidden = model.init_hidden(batch_size=1)  # batch_size might vary depending on your use case
            #else:
                #hidden=hidden.detach() #you detach if you want to avoid that the gradient is propagated through the hidden states to avoid long training time and memory usage
            
            # Forward pass for prediction
            output, hidden = model(batch_x, rounds)
            prediction = torch.round(output)  # Convert probability to binary (0 or 1)
            
            for j in range(0,batch_size):
                if prediction[j] == target[j]:
                    correct += 1
    
    accuracy = correct / len(test_sequences)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

####################################################################################################################

# Hyperparameters
input_size = 1  # Each Lattice RNN cell takes 1 bit as input
hidden_size = 128 # Hidden size of each RNN cell
output_size = 1  # Binary output (e.g., 0 or 1)
grid_height = 4  # Number of rows in the grid
grid_width = 2   # Number of columns in the grid
learning_rate = 0.001
num_epochs = 20
batch_size = 1024
layers_sizes=[hidden_size*3,hidden_size*2,hidden_size ]

print(f'2D LSTM_batch pre')
print(f'circuit_google, rounds={rounds}, distance = {distance} num_shots={num_shots}, batch_size = {batch_size}, hidden_size = {hidden_size}, batch_size = {batch_size}, layers_sizes={layers_sizes},  learning_rate={learning_rate}, num_epochs={num_epochs}')

#########################################################################################################

# Create a model instance
model = BlockRNN(input_size, hidden_size, output_size, grid_height, grid_width, rounds,layers_sizes,batch_size)

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary cross-entropy loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

test_size=0.2
test_dataset_size=num_shots*test_size
X_train, X_test, y_train, y_test = train_test_split(detection_array1, observable_flips, test_size=0.2, random_state=42, shuffle=False)

# Training the model
train_rnn(model, X_train, y_train, criterion, optimizer, num_epochs,batch_size,rounds)


test(model, X_test, y_test,batch_size)

# Assuming `model` is your trained LSTM model
torch.save(model.state_dict(), "2D_LSTM_pre_r5.pth")

#############################################################################################################

"""
# Reinitialize the model architecture
model = BlockRNN(input_size, hidden_size, output_size, grid_height, grid_width, rounds,layers_sizes,batch_size)

# Load the saved weights
model.load_state_dict(torch.load("lstm_model.pth"))

# Training the model
finetune_rnn(model, X_train, y_train, criterion, optimizer, num_epochs,batch_size,rounds)

X_train, X_test, y_train, y_test = train_test_split(detection_array1, observable_flips, test_size=0.2, random_state=42, shuffle=False)


test(model, X_test, y_test,batch_size)
"""

