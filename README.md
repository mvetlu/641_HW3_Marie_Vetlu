Student: Marie Vetluzhskikh
\
Class: MSML641 PCS4\
Date: Nov 12, 2025
# Setup and Installation

#### A. System Requirements
* **Python Version:** Python 3.9 or higher
* **Operating System:** Developed and tested on macOS (Intel Core i9)
#### B. Hardware Used for Reproducibility
The runtime metrics in this report were generated using the following hardware:

| Category         | Specification                    |
| :--------------- | :------------------------------- |
| **Model**        | **MacBook Pro (16-inch, 2019)**  |
| **Processor**    | **2.3 GHz 8-Core Intel Core i9** |
| **Memory (RAM)** | **16 GB 2667 MHz DDR4**          |
| **Device Used**  | CPU Only                         |
#### C. Installation Steps

1.  **Clone the repository:**
```bash
git clone https://github.com/mvetlu/641_HW3_Marie_Vetlu.git
cd 641_HW3_Marie_Vetlu
```

2. **Create and activate a virtual environment (Mac)** 
```bash
python3 -m venv venv
source venv/bin/activate
```

3.  **Install Dependencies:**
All required dependencies (TensorFlow/Keras, NumPy, Pandas, Scikit-learn, etc.) are listed in `requirements.txt`
```bash
pip install -r requirements.txt
```

## How Run the Scripts

The entire pipeline consists of preprocessing, training, and evaluation.

**Note:** The results files (`.keras` models, `.csv` metrics, and plots) are already included in the `results/` folder. The steps below are for full verification.
#### Preprocessing (`preprocess.py`)

This step tokenizes and pads the data to the required sequence lengths (25, 50, 100). **(!) Run this only once (!)**

```bash
cd src/
python preprocess.py
```

#### Training Models (`train.py`)

This script trains 68 models and saves their weights and training histories.

My code uses a maximum of 10 epochs with Early Stopping (patience=3) to prevent overfitting and an SGD learning rate of 0.2 to ensure convergence.

Trained model files (`.keras`) are saved to `../results/models/`
To train the models for the first time instead of using the saved ones:
```bash
cd src/
python train.py
```

#### Evaluation (`evaluate.py`)

This script loads the trained models, calculates the final test metrics (Accuracy, F1-score), and generates all analysis plots and summary statistics.

```bash
cd src/
python evaluate.py
```

## Expected Runtime and Output Artifacts

#### Expected Runtime

The total time required to run all 68 training experiments to completion (up to 10 epochs each) on the specified hardware is:

| Step            | Estimated Runtime (on specified hardware)                      |
| --------------- |----------------------------------------------------------------|
| `preprocess.py` | 1-2 minutes                                                    |
| `train.py`      | ~5.3 min/model if 10 epochs;<br/>4-5.5 hours for all 68 models |
| `evaluate.py`   | 7-10 minutes                                                   |
#### Key Output Files

The final results are saved in the `results/` directory and can be inspected without re-running the training script.

| Output                          | Purpose                                                                                                                                                    |
| ------------------------------- |------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`results/metrics.csv`**       | primary report table: final test accuracy, F1-score, loss, and time per epoch (s) for all 68 experiments                                                   |
| **`results/summary_stats.txt`** | quick summary of the best model and overall project statistics                                                                                             |
| **`results/plots/`**            | All necessary visualizations, including individual training history plots, comparison charts (e.g., accuracy vs. architecture), and the Top 5 models chart |
| **`results/models/`**           | 68 trained model weights (`.keras` files)                                                                                                                  |
