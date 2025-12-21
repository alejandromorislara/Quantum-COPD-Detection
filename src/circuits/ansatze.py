"""
Variational ansatz circuits for quantum machine learning.
"""
import pennylane as qml
from pennylane import numpy as np
from typing import Optional


def strongly_entangling_ansatz(weights: np.ndarray, wires: list) -> None:
    """
    Strongly entangling layers ansatz.
    
    Each layer consists of:
    - Single-qubit rotations (Rot gates with 3 parameters each)
    - CNOT entanglement with varying patterns
    
    Args:
        weights: Trainable parameters (shape from qml.StronglyEntanglingLayers.shape)
        wires: List of qubit wires
    """
    qml.StronglyEntanglingLayers(weights, wires=wires)


def basic_entangler_ansatz(weights: np.ndarray, wires: list, 
                           rotation: str = "Y") -> None:
    """
    Basic entangler layers ansatz.
    
    Each layer consists of:
    - Single-qubit rotation gates
    - CNOT entanglement in linear chain
    
    Args:
        weights: Trainable parameters (shape: n_layers x n_qubits)
        wires: List of qubit wires
        rotation: Type of rotation gate ("X", "Y", or "Z")
    """
    qml.BasicEntanglerLayers(weights, wires=wires, rotation=rotation)


def hardware_efficient_ansatz(weights: np.ndarray, wires: list, 
                              n_layers: int) -> None:
    """
    Hardware-efficient ansatz suitable for NISQ devices.
    
    Uses only RY and RZ rotations with CNOT entanglement.
    
    Args:
        weights: Trainable parameters (shape: n_layers x n_qubits x 2)
        wires: List of qubit wires
        n_layers: Number of layers
    """
    n_qubits = len(wires)
    
    for layer in range(n_layers):
        # Single-qubit rotations
        for i, wire in enumerate(wires):
            qml.RY(weights[layer, i, 0], wires=wire)
            qml.RZ(weights[layer, i, 1], wires=wire)
        
        # Entanglement (linear chain)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])


def simplified_two_design_ansatz(weights: np.ndarray, wires: list,
                                  n_layers: int) -> None:
    """
    Simplified Two-Design ansatz.
    
    Provides good expressibility with minimal gate count.
    
    Args:
        weights: Trainable parameters (shape: n_layers x n_qubits x 2)
        wires: List of qubit wires
        n_layers: Number of layers
    """
    n_qubits = len(wires)
    
    for layer in range(n_layers):
        # RY rotations
        for i, wire in enumerate(wires):
            qml.RY(weights[layer, i, 0], wires=wire)
        
        # CZ entanglement
        for i in range(0, n_qubits - 1, 2):
            qml.CZ(wires=[wires[i], wires[i + 1]])
        
        # More RY rotations
        for i, wire in enumerate(wires):
            qml.RY(weights[layer, i, 1], wires=wire)
        
        # Shifted CZ entanglement
        for i in range(1, n_qubits - 1, 2):
            qml.CZ(wires=[wires[i], wires[i + 1]])


def tree_tensor_ansatz(weights: np.ndarray, wires: list) -> None:
    """
    Tree tensor network ansatz.
    
    Creates a hierarchical entanglement structure.
    
    Args:
        weights: Trainable parameters
        wires: List of qubit wires (must be power of 2)
    """
    n_qubits = len(wires)
    layer_idx = 0
    
    # Initial rotations
    for i, wire in enumerate(wires):
        qml.RY(weights[layer_idx, i], wires=wire)
    layer_idx += 1
    
    # Tree structure
    step = 1
    while step < n_qubits:
        for i in range(0, n_qubits, 2 * step):
            if i + step < n_qubits:
                qml.CNOT(wires=[wires[i], wires[i + step]])
                qml.RY(weights[layer_idx, i // (2 * step)], wires=wires[i + step])
        layer_idx += 1
        step *= 2


def get_ansatz_shape(ansatz_type: str, n_qubits: int, n_layers: int) -> tuple:
    """
    Get the weight shape for a given ansatz type.
    
    Args:
        ansatz_type: Type of ansatz
        n_qubits: Number of qubits
        n_layers: Number of layers
        
    Returns:
        Shape tuple for the weights array
    """
    if ansatz_type == "strongly_entangling":
        return qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)
    elif ansatz_type == "basic_entangler":
        return (n_layers, n_qubits)
    elif ansatz_type == "hardware_efficient":
        return (n_layers, n_qubits, 2)
    elif ansatz_type == "simplified_two_design":
        return (n_layers, n_qubits, 2)
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz_type}")

