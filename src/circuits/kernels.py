"""
Quantum kernel functions for QSVM.
"""
import pennylane as qml
from pennylane import numpy as np
from typing import Callable, Optional
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import N_QUBITS


def create_quantum_kernel(n_qubits: int = N_QUBITS,
                          feature_map: str = "custom") -> Callable:
    """
    Create a quantum kernel function.
    
    Args:
        n_qubits: Number of qubits
        feature_map: Type of feature map ("angle", "iqp", "custom", "zz")
        
    Returns:
        Quantum kernel function
    """
    dev = qml.device("default.qubit", wires=n_qubits)
    wires = list(range(n_qubits))
    
    def feature_map_circuit(x):
        """Apply the chosen feature map."""
        if feature_map == "angle":
            qml.AngleEmbedding(x, wires=wires)
        elif feature_map == "iqp":
            qml.IQPEmbedding(x, wires=wires)
        elif feature_map == "custom":
            # Custom feature map with more expressivity
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(x[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            for i in range(n_qubits):
                qml.RY(x[i], wires=i)
        elif feature_map == "zz":
            # ZZ feature map
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(2 * x[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(2 * x[i] * x[i + 1], wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
        else:
            raise ValueError(f"Unknown feature map: {feature_map}")
    
    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        """
        Compute quantum kernel K(x1, x2) = |<φ(x1)|φ(x2)>|^2
        
        This is computed as the probability of measuring |00...0⟩
        after applying φ(x1)† φ(x2).
        """
        feature_map_circuit(x1)
        qml.adjoint(feature_map_circuit)(x2)
        return qml.probs(wires=wires)
    
    def quantum_kernel(x1, x2):
        """Compute kernel value between two data points."""
        probs = kernel_circuit(x1, x2)
        return probs[0]  # Probability of |00...0⟩
    
    return quantum_kernel


def kernel_matrix(X1: np.ndarray, X2: np.ndarray,
                  kernel_fn: Callable,
                  show_progress: bool = True) -> np.ndarray:
    """
    Compute the kernel matrix between two sets of data points.
    
    Args:
        X1: First set of data points (n1 x d)
        X2: Second set of data points (n2 x d)
        kernel_fn: Quantum kernel function
        show_progress: Whether to show progress bar
        
    Returns:
        Kernel matrix (n1 x n2)
    """
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))
    
    total = n1 * n2
    iterator = range(n1)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing kernel matrix")
    
    for i in iterator:
        for j in range(n2):
            K[i, j] = kernel_fn(X1[i], X2[j])
    
    return K


def kernel_matrix_symmetric(X: np.ndarray,
                           kernel_fn: Callable,
                           show_progress: bool = True) -> np.ndarray:
    """
    Compute symmetric kernel matrix (for training set).
    
    More efficient than kernel_matrix when X1 == X2 because
    K[i,j] = K[j,i].
    
    Args:
        X: Data points (n x d)
        kernel_fn: Quantum kernel function
        show_progress: Whether to show progress bar
        
    Returns:
        Symmetric kernel matrix (n x n)
    """
    n = len(X)
    K = np.zeros((n, n))
    
    # Compute only upper triangle
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    iterator = pairs
    if show_progress:
        iterator = tqdm(pairs, desc="Computing kernel matrix")
    
    for i, j in iterator:
        k_val = kernel_fn(X[i], X[j])
        K[i, j] = k_val
        K[j, i] = k_val  # Symmetry
    
    return K


class QuantumKernel:
    """
    Quantum kernel class for use with sklearn SVM.
    
    Provides a sklearn-compatible interface for quantum kernels.
    """
    
    def __init__(self, n_qubits: int = N_QUBITS,
                 feature_map: str = "custom"):
        """
        Initialize the quantum kernel.
        
        Args:
            n_qubits: Number of qubits
            feature_map: Type of feature map
        """
        self.n_qubits = n_qubits
        self.feature_map = feature_map
        self.kernel_fn = create_quantum_kernel(n_qubits, feature_map)
        self._X_train = None
        
    def fit(self, X: np.ndarray) -> 'QuantumKernel':
        """Store training data for later kernel computation."""
        self._X_train = X
        return self
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Compute kernel matrix (sklearn interface).
        
        Args:
            X1: First set of data points
            X2: Second set of data points
            
        Returns:
            Kernel matrix
        """
        return kernel_matrix(X1, X2, self.kernel_fn, show_progress=False)
    
    def compute_train_kernel(self, show_progress: bool = True) -> np.ndarray:
        """Compute kernel matrix for training data."""
        if self._X_train is None:
            raise ValueError("Must call fit() first")
        return kernel_matrix_symmetric(self._X_train, self.kernel_fn, show_progress)
    
    def compute_test_kernel(self, X_test: np.ndarray,
                           show_progress: bool = True) -> np.ndarray:
        """Compute kernel matrix between test and training data."""
        if self._X_train is None:
            raise ValueError("Must call fit() first")
        return kernel_matrix(X_test, self._X_train, self.kernel_fn, show_progress)


def quantum_kernel(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Default quantum kernel function.
    
    Wrapper for backward compatibility.
    """
    kernel_fn = create_quantum_kernel()
    return kernel_fn(x1, x2)

