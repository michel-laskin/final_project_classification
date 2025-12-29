"""
Simple example demonstrating windowing-based feature extraction
"""

# Fix for OpenMP duplicate library warning
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from main import HRVPipeline
from pathlib import Path

def main():
    """Run the pipeline with windowing (default behavior)"""
    
    # Find a CSV file in recordings folder
    recordings_path = Path(__file__).parent / "recordings"
    csv_files = list(recordings_path.glob("*.csv"))
    
    if len(csv_files) == 0:
        print("No CSV files found in recordings folder")
        return
    
    csv_file = csv_files[0]
    print(f"Processing: {csv_file.name}\n")
    
    # Initialize pipeline with custom windowing parameters
    pipeline = HRVPipeline(config={
        "window_size_sec": 10,      # 10 second windows
        "window_overlap": 0.5,       # 50% overlap
        "min_peaks_per_window": 3    # At least 3 peaks per window
    })
    
    # Run the complete pipeline
    # This will:
    # 1. Load the CSV data
    # 2. Preprocess (filter and detect peaks)
    # 3. Create overlapping windows
    # 4. Extract features from each window
    # 5. Encode features
    # 6. Process through TCN
    results = pipeline.run_pipeline(csv_file, plot=True)
    
    # Access results
    print("\n" + "="*70)
    print("Results:")
    print("="*70)
    print(f"Features tensor shape: {results['features_tensor'].shape}")
    print(f"  - Dimensions: [num_windows, 1, num_features]")
    print(f"  - Number of windows: {results['features_tensor'].shape[0]}")
    print(f"  - Features per window: {results['features_tensor'].shape[2]}")
    
    print(f"\nEncoded features shape: {results['encoded_features'].shape}")
    print(f"TCN output shape: {results['tcn_output'].shape}")

if __name__ == "__main__":
    main()
