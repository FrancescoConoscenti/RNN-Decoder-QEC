import stim
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List


distance=3
rounds=21

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

surface_code_circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_x",
    rounds=rounds,
    distance=distance,
    after_clifford_depolarization=0.01,
    after_reset_flip_probability=0.01,
    before_measure_flip_probability=0.01,
    before_round_data_depolarization=0.01)

####################################################################################################################
#get synthetic data
num_shots=20000000
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

test_size=0.2
test_dataset_size=num_shots*test_size
X_train, X_test, y_train, y_test = train_test_split(detection_array1, observable_flips, test_size=0.2, random_state=42, shuffle=False)

###################################################################################################################
#experimental
"""def parse_b8(data: bytes, bits_per_shot: int) -> List[List[bool]]:
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


path1 = r"google_qec3v5_experiment_data/surface_code_bX_d3_r05_center_3_5/detection_events.b8"
path2 = r"google_qec3v5_experiment_data/surface_code_bX_d3_r05_center_3_5/obs_flips_actual.01"
round = 5
bits_per_shot = round*8

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
"""
##############################################################################################################################
#train model
import torch
import torch.nn as nn
import torch.optim as optim


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNetwork, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)  # Hidden layer
        self.relu = nn.ReLU()                             # ReLU activation
        self.output = nn.Linear(hidden_size, output_size)  # Output layer
        self.sigmoid = nn.Sigmoid()                       # Sigmoid activation for output

    def forward(self, x):
        x = self.hidden(x)            # Pass input through hidden layer
        x = self.relu(x)              # Apply ReLU activation
        x = self.output(x)            # Pass through output layer
        x = self.sigmoid(x)           # Apply sigmoid activation
        return x

# RNN model
class BinaryRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,num_layers,hidden_layer_Ff):
        super(BinaryRNN, self).__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Linear(input_size, input_size)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.Ff = FeedForwardNetwork(hidden_size, hidden_layer_Ff, output_size)
        self.sigmoid = nn.Sigmoid()  # For binary output
    
    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.Ff(out[:, -1, :])  # Use the last time-step's output, needed for changing the dimension of the output compared of input
        out = self.sigmoid(out)  # I need a Binary output
        return out, hidden
    
    def init_hidden(self, batch_size,num_layers):
        return torch.zeros(num_layers, batch_size, self.hidden_size)

# LSTM model    
class BinaryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,num_layers, hidden_layer_Ff,dropout):
        super(BinaryLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size,num_layers, dropout=0.2, batch_first=True)
        self.relu=nn.ReLU()
        self.Ff1 = FeedForwardNetwork(hidden_size, hidden_layer_Ff, output_size)
        self.Ff2 = FeedForwardNetwork(hidden_size+input_size, hidden_layer_Ff, output_size)
        self.sigmoid = nn.Sigmoid()  # For binary output
    
    def forward(self, x):
        out, hidden = self.lstm(x[:, :-2, :])
        out1=self.relu(out[:, -1, :]) # Use the last time-step's output, needed for changing the dimension of the output compared of input
        p_aux = self.Ff1(out1)  
        
        last_defect=x[:,-1, :]
        out2 = torch.cat((out1, last_defect),dim=1)
        p_main= self.Ff2(out2)
        return p_main, p_aux
    
    def init_hidden(self, batch_size,num_layers):
        return (torch.zeros(num_layers, batch_size, self.hidden_size),  # Hidden state
                torch.zeros(num_layers, batch_size, self.hidden_size))  # Cell state



# Function to convert binary string to tensor
def binary_array_to_tensor(binary_array):
    # Check if the input is a NumPy array, if not, convert it
    if isinstance(binary_array, np.ndarray):
        tensor = torch.from_numpy(binary_array).float()  # Convert NumPy array to float32 tensor
    else:
        # If not a NumPy array, convert it as before
        tensor = torch.tensor([[int(bit) for bit in binary_array]], dtype=torch.float32)
    return tensor

def H(p_i,p_j):

    res=-1*p_i*torch.log(p_j)-(1-p_i)*torch.log(1-p_j)

    return res

# Training function
def train(model, binary_sequences, targets, num_epochs, learning_rate, batch_size,num_layers):
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Calculate number of batches
    num_batches = len(binary_sequences) // batch_size
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch_idx in range(num_batches):
            # Get batch data
            batch_sequences = binary_sequences[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            batch_targets = targets[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            target_tensor = torch.tensor(batch_targets).float()
            
            optimizer.zero_grad()
            
            # Initialize hidden state for the batch with batch size
            hidden = model.init_hidden(batch_size,num_layers)
            
            # Forward pass through each sequence in the batch
            batch_loss = 0

            input_tensor = binary_array_to_tensor(batch_sequences)  # Prepare input tensor for batch
            p_main, p_aux = model(input_tensor)  # Forward pass
            p_main, p_aux=p_main.squeeze(1), p_aux.squeeze(1)
            
            loss= H(target_tensor,p_main)+0.5*H(target_tensor,p_aux)
            #loss = criterion(p_main, target_tensor)
            loss=loss.mean()
            
            batch_loss += loss.mean().item()
            
            # Compute the average loss for the batch
            batch_loss = batch_loss / batch_size
            total_loss += batch_loss
            
            # Backward pass and optimization step
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')
    print("End of training")


def finetune(model, binary_sequences, targets, num_epochs, learning_rate, batch_size,num_layers):
    
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Calculate number of batches
    num_batches = len(binary_sequences) // batch_size
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch_idx in range(num_batches):
            # Get batch data
            batch_sequences = binary_sequences[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            batch_targets = targets[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            target_tensor = torch.tensor(batch_targets).float()
            
            optimizer.zero_grad()
            
            # Initialize hidden state for the batch with batch size
            hidden = model.init_hidden(batch_size,num_layers)
            
            # Forward pass through each sequence in the batch
            batch_loss = 0

            input_tensor = binary_array_to_tensor(batch_sequences)  # Prepare input tensor for batch
            p_main, p_aux = model(input_tensor)  # Forward pass
            p_main, p_aux=p_main.squeeze(1), p_aux.squeeze(1)
            
            loss= H(target_tensor,p_main)+0.5*H(target_tensor,p_aux)
            #loss = criterion(p_main, target_tensor)
            loss=loss.mean()
            
            batch_loss += loss.mean().item()
            
            # Compute the average loss for the batch
            batch_loss = batch_loss / batch_size
            total_loss += batch_loss
            
            # Backward pass and optimization step
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')
    print("End of finetuning")



def test(model, binary_sequences, targets, batch_size,num_layers):
    model.eval()  # Set the model to evaluation mode (disable dropout, etc.)
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient computation for testing
        # Calculate number of batches
        num_batches = len(binary_sequences) // batch_size
        
        for batch_idx in range(num_batches):
            # Get batch data
            batch_sequences = binary_sequences[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            batch_targets = targets[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            
            # Convert batch sequences to a tensor
            input_tensor = torch.stack([binary_array_to_tensor(seq) for seq in batch_sequences])  # Shape: (batch_size, seq_length, input_size)
            
            # Initialize hidden state for the batch
            hidden = model.init_hidden(batch_size, num_layers)
            
            # Forward pass
            outputs, hidden = model(input_tensor)
            
            # Convert outputs to binary predictions (0 or 1)
            predictions = torch.round(outputs.squeeze()).int()  # Convert probabilities to binary
            
            # Check predictions against targets
            for pred, target in zip(predictions, batch_targets):
                if pred.item() == target:
                    correct += 1
                total += 1

        # Calculate accuracy
        accuracy = correct / total
        print(f'Test Accuracy: {accuracy * 100:.2f}%')

########################################################################################################################à
# Define parameters
input_size = num_ancilla_qubits # Each input is a Detection round, vector of mmt of the Ancilla
hidden_size = 64  # You can experiment with different sizes
output_size = 1  # Output is the value of the observable after the mmt cycles
batch_size=256

learning_rate=0.00025
learning_rate_fine=0.000025
num_epochs=20
num_epochs_fine=5

num_layers=2
hidden_layer_Ff=64
dropout=0.2

print(f'LSTM_Teheral')
print(f'circuit_google, rounds={rounds}, distance = {distance} num_shots={num_shots}, hidden_size = {hidden_size}, batch_size = {batch_size}, num_layers={num_layers}, hidden_layer_Ff={hidden_layer_Ff},  learning_rate={learning_rate}, num_epochs={num_epochs}')


# Create an instance of the RNN model
model = BinaryLSTM(input_size, hidden_size, output_size,num_layers,hidden_layer_Ff,dropout)

# Train the model
train(model, X_train, y_train, num_epochs, learning_rate, batch_size,num_layers)

#finetune(model, X_train_exp, y_train_exp, num_epochs_fine, learning_rate_fine, batch_size,num_layers)
    
#test(model, X_test_exp, y_test_exp, batch_size, num_layers)
test(model, X_test, y_test, batch_size, num_layers)

import pymatching

detector_error_model = circuit_google.detector_error_model(decompose_errors=True)
matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

detection_test=detection_events[-int(test_dataset_size):]
observable_test=observable_flips[-int(test_dataset_size):]


# Run the decoder with the test samples
predictions = matcher.decode_batch(detection_test)

# Count the mistakes.
num_errors = 0
for shot in range(int(test_dataset_size)):
    actual_for_shot = observable_test[shot]
    predicted_for_shot = predictions[shot][0]
    if not np.array_equal(actual_for_shot, predicted_for_shot):
        num_errors += 1


print(f'Accuracy MWPM{(test_dataset_size-num_errors)/test_dataset_size}')