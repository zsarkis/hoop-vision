"""Stage 5: Prediction - Train and evaluate prediction models."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

from .config import Config


# Prediction target classes
OUTCOME_CLASSES = [
    "made_2pt",
    "made_3pt",
    "missed_shot",
    "turnover",
    "foul_offensive",
    "foul_defensive",
    "end_period"
]


class PredictionModel:
    """Train and evaluate prediction models for next possession outcomes."""

    def __init__(self, config: Config):
        """
        Initialize prediction model.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.output_dir = config.predictions_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_save_dir = config.model_save_dir
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

        self.model = None

    def load_all_features(self) -> pd.DataFrame:
        """
        Load features from all processed clips.

        Returns:
            Combined DataFrame with all features
        """
        features_dir = self.config.features_output_dir
        if not features_dir.exists():
            raise ValueError(f"Features directory not found: {features_dir}")

        all_features = []
        clip_dirs = [d for d in features_dir.iterdir() if d.is_dir()]

        for clip_dir in clip_dirs:
            features_file = clip_dir / "features.csv"
            if features_file.exists():
                df = pd.read_csv(features_file)
                df["clip_id"] = clip_dir.name
                all_features.append(df)

        if not all_features:
            raise ValueError("No feature files found")

        combined_df = pd.concat(all_features, ignore_index=True)
        print(f"Loaded features from {len(clip_dirs)} clips: {len(combined_df)} samples")

        return combined_df

    def load_labels(self, labels_file: Optional[Path] = None) -> pd.DataFrame:
        """
        Load ground truth labels for clips.

        Args:
            labels_file: Path to CSV with clip_id, frame_idx, label columns

        Returns:
            DataFrame with labels
        """
        if labels_file is None:
            labels_file = Path("data/labels.csv")

        if not labels_file.exists():
            print(f"Warning: Labels file not found at {labels_file}")
            print("Creating dummy labels for demonstration purposes")
            # Create dummy labels - in real usage, these must be manually annotated
            features_df = self.load_all_features()
            dummy_labels = pd.DataFrame({
                "clip_id": features_df["clip_id"],
                "frame_idx": features_df["frame_idx"],
                "label": np.random.choice(OUTCOME_CLASSES, size=len(features_df))
            })
            return dummy_labels

        return pd.read_csv(labels_file)

    def prepare_dataset(self, features_df: pd.DataFrame, labels_df: pd.DataFrame):
        """
        Merge features and labels, prepare for training.

        Args:
            features_df: Features DataFrame
            labels_df: Labels DataFrame

        Returns:
            Tuple of (X, y, feature_columns)
        """
        # Merge features and labels
        df = features_df.merge(labels_df, on=["clip_id", "frame_idx"], how="inner")
        print(f"Merged dataset: {len(df)} labeled samples")

        # Separate features and labels
        feature_columns = [col for col in df.columns
                          if col not in ["clip_id", "frame_idx", "timestamp", "label"]]

        X = df[feature_columns].fillna(0)  # Fill NaN with 0
        y = df["label"]

        print(f"Features: {len(feature_columns)} columns")
        print(f"Label distribution:\n{y.value_counts()}")

        return X, y, feature_columns

    def train_baseline(self, labels_file: Optional[Path] = None) -> Dict[str, Any]:
        """
        Train baseline XGBoost model.

        Args:
            labels_file: Optional path to labels CSV

        Returns:
            Dictionary with model path and metrics
        """
        print("Training baseline XGBoost model")

        # Load data
        features_df = self.load_all_features()
        labels_df = self.load_labels(labels_file)
        X, y, feature_columns = self.prepare_dataset(features_df, labels_df)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_split,
            random_state=self.config.random_seed,
            stratify=y if len(y.unique()) > 1 else None
        )

        print(f"Train set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.config.random_seed,
            eval_metric="mlogloss"
        )

        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        print(f"\nBaseline Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Save model
        model_path = self.model_save_dir / "xgboost_baseline.json"
        self.model.save_model(model_path)
        print(f"\nModel saved to: {model_path}")

        # Save metrics
        metrics = {
            "accuracy": accuracy,
            "classification_report": report,
            "num_train_samples": len(X_train),
            "num_test_samples": len(X_test),
            "num_features": len(feature_columns),
            "feature_columns": feature_columns
        }

        metrics_path = self.output_dir / "baseline_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Metrics saved to: {metrics_path}")

        return {
            "model_path": str(model_path),
            "metrics": metrics
        }


def main():
    """Run prediction model training."""
    config = Config()
    model = PredictionModel(config)
    model.train_baseline()


if __name__ == "__main__":
    main()
