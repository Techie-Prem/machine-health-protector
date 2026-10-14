# Machine Health Protector (Predictive Maintenance ML)

An end-to-end Machine Learning web application designed to forecast industrial machinery failures and assess operational health before breakdowns occur. The solution features dimensionality reduction using PCA, feature scaling, classification modeling, and an interactive Streamlit dashboard.

---

## Key Features

- **Failure Prediction:** Predicts potential machine failures based on sensor readings and operational metrics.
- **Dimensionality Reduction:** Incorporates Principal Component Analysis (PCA) to extract key variance components from multi-sensor data.
- **Interactive UI:** Web-based interface built with Streamlit for single-sample input testing and real-time failure risk analysis.
- **Container Ready:** Preconfigured `.devcontainer` settings for containerized development workflows.

---

## Project Structure

```text
machine-health-predictor/
├── .devcontainer/
│   └── devcontainer.json      # Dev container setup
├── app.py                     # Streamlit frontend application
├── model.py                   # Model pipeline and inference logic
├── model_metadata.json        # Training parameters and model metadata
├── pca.pkl                    # Fitted PCA transformer
├── scaler.pkl                 # Fitted feature scaler (e.g., StandardScaler)
├── trained_model.pkl          # Serialized predictive ML model
├── requirements.txt           # Project dependencies
├── runtime.txt                # Target Python runtime environment
└── README.md                  # Project documentation
Installation & Setup
1. Clone the Repository
Bash
git clone [https://github.com/your-username/machine-health-predictor.git](https://github.com/your-username/machine-health-predictor.git)
cd machine-health-predictor
2. Set Up a Virtual Environment
Bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
Bash
pip install -r requirements.txt
Usage
Run the Streamlit Dashboard
Launch the web interface locally:

Bash
streamlit run app.py
After running the command, open your browser and navigate to:

Plaintext
http://localhost:8501
Run Model Inference via Terminal
To test or trigger the model directly from Python:

Bash
python model.py
Machine Learning Pipeline
Pre-processing: Raw inputs and operational sensor features are normalized using scaler.pkl.

Feature Transformation: pca.pkl reduces multicollinearity and projects input vectors into lower-dimensional principal components.

Inference: trained_model.pkl evaluates the transformed input to output the probability and type of machinery breakdown.

Metadata: Model versions, schema mappings, and metric summaries are recorded in model_metadata.json.

Tech Stack
Language: Python

Libraries: Streamlit, Scikit-learn, Pandas, NumPy, Joblib/Pickle

Development Tools: VS Code Dev Containers, Git
