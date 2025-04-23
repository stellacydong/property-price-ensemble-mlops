# property-price-ensemble-mlops

A repository for building, evaluating, and deploying an ensemble model for property price prediction, complete with an MLOps workflow using MLflow.

## Repository Structure

```
property-price-ensemble-mlops/
├── data/
│   └── train_ml_ops.csv       # Training dataset for model development
├── notebooks/
│   └── property1.ipynb        # Exploratory analysis and prototyping notebook
├── src/
│   ├── property1.py           # Data processing and feature engineering scripts
│   └── train_and_log.py       # Training script that logs parameters, metrics, and models with MLflow
└── README.md                  # Project overview and instructions
```  

## Getting Started

### Prerequisites

- Python 3.8+  
- (Optional) [virtualenv](https://virtualenv.pypa.io/) or [conda](https://docs.conda.io/) for environment isolation

### Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/property-price-ensemble-mlops.git
   cd property-price-ensemble-mlops
   ```

2. **Create and activate a virtual environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate       # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

> **Note:** Populate `requirements.txt` with your project dependencies (e.g., `pandas`, `scikit-learn`, `numpy`, `xgboost`, `lightgbm`, `mlflow`).

## Usage

### 1. Exploratory Analysis

Open the Jupyter notebook to explore the dataset and prototype models:  
```bash
jupyter notebook notebooks/property1.ipynb
```

### 2. Model Training & Logging

Run the training script to train the ensemble and log experiments with MLflow:  
```bash
python src/train_and_log.py \
  --data_path data/train_ml_ops.csv \
  --experiment_name "Property_Price_Ensemble"
```

### 3. Viewing MLflow UI

Start the MLflow tracking UI to inspect runs, metrics, and artifacts:  
```bash
mlflow ui
```
Then navigate to `http://localhost:5000` in your browser.

## File Descriptions

- **data/train_ml_ops.csv**: The cleaned and prepared dataset used for training and evaluation.
- **notebooks/property1.ipynb**: Contains data exploration, visualization, and initial model prototyping steps.
- **src/property1.py**: Script for data loading, cleaning, feature engineering, and preprocessing.
- **src/train_and_log.py**: Orchestrates model training for a voting ensemble (LightGBM, HistGB, XGBoost), logs parameters, metrics, and artifacts to MLflow.

## Contributing

Feel free to open issues or submit pull requests for improvements.  

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

