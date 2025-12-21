"""
Audio feature extraction for respiratory sounds using MFCCs.
"""
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from typing import List, Optional, Tuple, Union
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import (
    SAMPLE_RATE, N_MFCC, HOP_LENGTH, N_FFT, 
    FEATURE_NAMES, TARGET_CLASSES
)
from config.paths import AUDIO_DIR, FEATURES_FILE, PATIENT_DIAGNOSIS_FILE


class AudioFeatureExtractor:
    """
    Extracts audio features (MFCCs and their deltas) from respiratory sound segments.
    
    Features extracted (52 total):
    - 13 MFCCs (mean)
    - 13 MFCCs (std)
    - 13 Delta MFCCs (mean)
    - 13 Delta MFCCs (std)
    """
    
    def __init__(self, n_mfcc: int = N_MFCC, 
                 sr: int = SAMPLE_RATE,
                 hop_length: int = HOP_LENGTH,
                 n_fft: int = N_FFT):
        """
        Initialize the feature extractor.
        
        Args:
            n_mfcc: Number of MFCC coefficients to extract
            sr: Sample rate for audio processing
            hop_length: Hop length for MFCC computation
            n_fft: FFT window size
        """
        self.n_mfcc = n_mfcc
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.feature_names = self._generate_feature_names()
        
    def _generate_feature_names(self) -> List[str]:
        """Generate feature names based on n_mfcc."""
        names = []
        for stat in ["mean", "std"]:
            for i in range(self.n_mfcc):
                names.append(f"mfcc_{i}_{stat}")
        for stat in ["mean", "std"]:
            for i in range(self.n_mfcc):
                names.append(f"delta_mfcc_{i}_{stat}")
        return names
    
    def extract_from_segment(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract features from a single audio segment (respiratory cycle).
        
        Args:
            audio: Audio signal as numpy array
            
        Returns:
            Feature vector (52 dimensions)
        """
        # Handle empty or very short segments
        if len(audio) < self.n_fft:
            # Pad with zeros if too short
            audio = np.pad(audio, (0, self.n_fft - len(audio)), mode='constant')
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=self.sr, 
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        
        # Compute delta MFCCs (first derivative)
        delta_mfccs = librosa.feature.delta(mfccs)
        
        # Compute statistics: mean and std for each coefficient
        features = np.concatenate([
            mfccs.mean(axis=1),         # 13 values: mean of each MFCC
            mfccs.std(axis=1),          # 13 values: std of each MFCC
            delta_mfccs.mean(axis=1),   # 13 values: mean of each delta MFCC
            delta_mfccs.std(axis=1)     # 13 values: std of each delta MFCC
        ])
        
        return features  # 52 dimensions total
    
    def extract_from_file(self, audio_path: Path) -> List[Tuple[np.ndarray, dict]]:
        """
        Extract features from all respiratory cycles in an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of tuples (features, metadata)
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sr)
        
        # Load annotations
        annotation_path = audio_path.with_suffix(".txt")
        if not annotation_path.exists():
            return []
        
        cycles = self._load_annotations(annotation_path)
        
        results = []
        for cycle in cycles:
            # Extract segment
            start_sample = int(cycle["start"] * sr)
            end_sample = int(cycle["end"] * sr)
            
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            if end_sample > start_sample:
                segment = audio[start_sample:end_sample]
                features = self.extract_from_segment(segment)
                
                results.append((features, cycle))
        
        return results
    
    def _load_annotations(self, annotation_path: Path) -> List[dict]:
        """Load respiratory cycle annotations from file."""
        cycles = []
        with open(annotation_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    cycles.append({
                        "start": float(parts[0]),
                        "end": float(parts[1]),
                        "crackles": int(parts[2]),
                        "wheezes": int(parts[3])
                    })
        return cycles
    
    def extract_from_dataframe(self, df: pd.DataFrame, 
                               audio_col: str = "audio") -> np.ndarray:
        """
        Extract features from audio segments stored in a DataFrame.
        
        Args:
            df: DataFrame with audio segments
            audio_col: Name of column containing audio arrays
            
        Returns:
            Feature matrix (n_samples x n_features)
        """
        features = []
        for audio in tqdm(df[audio_col], desc="Extracting features"):
            features.append(self.extract_from_segment(audio))
        
        return np.array(features)
    
    def extract_dataset(self, audio_dir: Optional[Path] = None,
                        diagnosis_file: Optional[Path] = None,
                        filter_classes: Optional[List[str]] = None,
                        save_path: Optional[Path] = None,
                        show_progress: bool = True) -> pd.DataFrame:
        """
        Extract features from all audio files in a directory.
        
        Args:
            audio_dir: Directory containing audio files
            diagnosis_file: Path to patient diagnosis CSV
            filter_classes: List of diagnosis classes to include
            save_path: Path to save the resulting DataFrame
            show_progress: Whether to show progress bar
            
        Returns:
            DataFrame with features and metadata
        """
        audio_dir = audio_dir or AUDIO_DIR
        diagnosis_file = diagnosis_file or PATIENT_DIAGNOSIS_FILE
        
        # Load patient diagnoses
        diagnosis_df = pd.read_csv(diagnosis_file, header=None,
                                   names=["patient_id", "diagnosis"])
        
        # Get audio files
        audio_files = sorted(audio_dir.glob("*.wav"))
        
        # Filter by diagnosis if specified
        if filter_classes:
            valid_patients = diagnosis_df[
                diagnosis_df["diagnosis"].isin(filter_classes)
            ]["patient_id"].values
            
            audio_files = [
                f for f in audio_files
                if int(f.stem.split("_")[0]) in valid_patients
            ]
        
        # Extract features
        all_data = []
        iterator = tqdm(audio_files, desc="Processing files") if show_progress else audio_files
        
        for audio_path in iterator:
            patient_id = int(audio_path.stem.split("_")[0])
            diagnosis = diagnosis_df[
                diagnosis_df["patient_id"] == patient_id
            ]["diagnosis"].values[0]
            
            # Extract features from all cycles in this file
            results = self.extract_from_file(audio_path)
            
            for features, cycle_info in results:
                row = {
                    "patient_id": patient_id,
                    "filename": audio_path.name,
                    "diagnosis": diagnosis,
                    "start": cycle_info["start"],
                    "end": cycle_info["end"],
                    "crackles": cycle_info["crackles"],
                    "wheezes": cycle_info["wheezes"],
                    "label": TARGET_CLASSES.get(diagnosis, -1)
                }
                
                # Add features
                for i, feat_name in enumerate(self.feature_names):
                    row[feat_name] = features[i]
                
                all_data.append(row)
        
        df = pd.DataFrame(all_data)
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"Features saved to {save_path}")
        
        return df
    
    def get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract feature matrix from a DataFrame with feature columns.
        
        Args:
            df: DataFrame with feature columns
            
        Returns:
            Feature matrix (n_samples x n_features)
        """
        return df[self.feature_names].values


def extract_features_pipeline(filter_classes: List[str] = ["Healthy", "COPD"],
                              save: bool = True) -> pd.DataFrame:
    """
    Complete feature extraction pipeline.
    
    Args:
        filter_classes: Classes to include in the dataset
        save: Whether to save the result to disk
        
    Returns:
        DataFrame with extracted features
    """
    extractor = AudioFeatureExtractor()
    
    save_path = FEATURES_FILE if save else None
    
    df = extractor.extract_dataset(
        filter_classes=filter_classes,
        save_path=save_path,
        show_progress=True
    )
    
    print(f"\nExtracted {len(df)} samples from {df['patient_id'].nunique()} patients")
    print(f"Class distribution: {df['diagnosis'].value_counts().to_dict()}")
    
    return df

