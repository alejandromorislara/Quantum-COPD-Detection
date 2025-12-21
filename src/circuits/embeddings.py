"""
Quantum embedding circuits for encoding classical data into quantum states.
"""
import pennylane as qml
from pennylane import numpy as np
from typing import Optional


def angle_embedding(x: np.ndarray, wires: list, rotation: str = "Y") -> None:
    """
    Encode classical data using angle embedding (rotation gates).
    
    Each feature is encoded as a rotation angle on a separate qubit.
    
    Args:
        x: Input features (length must equal number of wires)
        wires: List of qubit wires to use
        rotation: Rotation axis ("X", "Y", or "Z")
    """
    qml.AngleEmbedding(x, wires=wires, rotation=rotation)


def amplitude_embedding(x: np.ndarray, wires: list, 
                       normalize: bool = True, 
                       pad_with: float = 0.0) -> None:
    """
    Encode classical data as amplitudes of a quantum state.
    
    Requires 2^n features for n qubits (or pads with zeros).
    
    Args:
        x: Input features
        wires: List of qubit wires to use
        normalize: Whether to normalize the input
        pad_with: Value to pad with if needed
    """
    qml.AmplitudeEmbedding(x, wires=wires, normalize=normalize, pad_with=pad_with)


def iqp_embedding(x: np.ndarray, wires: list, n_repeats: int = 1) -> None:
    """
    IQP-style (Instantaneous Quantum Polynomial) embedding.
    
    Creates higher-order feature interactions through entanglement.
    
    Args:
        x: Input features
        wires: List of qubit wires
        n_repeats: Number of repetitions
    """
    qml.IQPEmbedding(x, wires=wires, n_repeats=n_repeats)


def custom_feature_map(x: np.ndarray, wires: list) -> None:
    """
    Custom feature map with Hadamard, rotations, and entanglement.
    
    This creates a more expressive feature space for kernel methods.
    
    Args:
        x: Input features (length must equal number of wires)
        wires: List of qubit wires to use
    """
    n_qubits = len(wires)
    
    # Layer 1: Hadamard + Z rotations
    for i, wire in enumerate(wires):
        qml.Hadamard(wires=wire)
        qml.RZ(x[i], wires=wire)
    
    # Entanglement layer
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
    
    # Layer 2: Y rotations
    for i, wire in enumerate(wires):
        qml.RY(x[i], wires=wire)


def zz_feature_map(x: np.ndarray, wires: list, reps: int = 2) -> None:
    """
    ZZ feature map with parameterized ZZ interactions.
    
    Similar to the ZZFeatureMap in Qiskit, creates entanglement
    through ZZ interactions with data-dependent parameters.
    
    Args:
        x: Input features
        wires: List of qubit wires
        reps: Number of repetitions
    """
    n_qubits = len(wires)
    
    for _ in range(reps):
        # Single-qubit rotations
        for i, wire in enumerate(wires):
            qml.Hadamard(wires=wire)
            qml.RZ(2 * x[i], wires=wire)
        
        # ZZ interactions (entanglement)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
            # ZZ interaction with product of features
            qml.RZ(2 * (np.pi - x[i]) * (np.pi - x[i + 1]), wires=wires[i + 1])
            qml.CNOT(wires=[wires[i], wires[i + 1]])


def data_reuploading_embedding(x: np.ndarray, weights: np.ndarray, 
                               wires: list, n_layers: int = 1) -> None:
    """
    Data re-uploading embedding where data is encoded multiple times.
    
    This is used in variational classifiers where data is interleaved
    with trainable parameters.
    
    Args:
        x: Input features
        weights: Trainable weights (shape: n_layers x n_qubits x 3)
        wires: List of qubit wires
        n_layers: Number of re-uploading layers
    """
    n_qubits = len(wires)
    
    for layer in range(n_layers):
        # Data encoding
        for i, wire in enumerate(wires):
            qml.RX(x[i % len(x)], wires=wire)
        
        # Trainable rotations
        for i, wire in enumerate(wires):
            qml.Rot(
                weights[layer, i, 0],
                weights[layer, i, 1],
                weights[layer, i, 2],
                wires=wire
            )
        
        # Entanglement
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
        if n_qubits > 1:
            qml.CNOT(wires=[wires[-1], wires[0]])  # Circular entanglement

