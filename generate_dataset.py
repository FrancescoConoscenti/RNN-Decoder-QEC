import stim
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

for rounds in (5,11,17):

    # Configuration parameters
    distance = 3
    num_shots = 10000000

    # Determine system size based on distance
    if distance == 3:
        num_qubits = 17
        num_data_qubits = 9
        num_ancilla_qubits = 8
    elif distance == 5:
        num_qubits = 49
        num_data_qubits = 25
        num_ancilla_qubits = 24

    if rounds == 5:
        path = r"google_qec3v5_experiment_data/surface_code_bX_d3_r05_center_3_5/circuit_noisy.stim"
    elif rounds == 11:
        path = r"google_qec3v5_experiment_data/surface_code_bX_d3_r11_center_3_5/circuit_noisy.stim"
    elif rounds == 17:
        path = r"google_qec3v5_experiment_data/surface_code_bX_d3_r17_center_3_5/circuit_noisy.stim"

    circuit_google = stim.Circuit.from_file(path)

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

    # Save with compression
    if rounds == 5:
        np.savez_compressed('data_stim/google_r5.npz', detection_array1 = detection_array1, observable_flips=observable_flips)
        print("saved round = 5")
    elif rounds == 11:
        np.savez_compressed('data_stim/google_r11.npz', detection_array1 = detection_array1, observable_flips=observable_flips)
        print("saved round = 11")
    elif rounds == 17:
        np.savez_compressed('data_stim/google_r17.npz', detection_array1 = detection_array1, observable_flips=observable_flips)
        print("saved round = 17")

        