# Credit Default Prediction - Streamlit App

A web application for predicting credit default risk using machine learning models.

## Features

- Interactive web interface built with Streamlit
- Support for multiple ML models:
  - Naive Bayes
  - Decision Tree
  - Support Vector Machine (SVM)
- Real-time predictions based on client financial data
- Docker support for easy deployment

## Input Variables

The application requires the following inputs:
- **Age**: Client's age (18-100 years)
- **Income**: Annual income in dollars
- **Loan Amount**: Total loan amount requested
- **Months Employed**: Number of months in current employment
- **Interest Rate**: Annual interest rate percentage

## Running Locally

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

3. Open your browser and navigate to `http://localhost:8501`

## Running with Docker

### Build the Docker image:
```bash
docker build -t credit-default-app .
```

### Run the container:
```bash
docker run -p 8501:8501 credit-default-app
```

### Access the application:
Open your browser and navigate to `http://localhost:8501`

## Docker Commands

### Stop the container:
```bash
docker stop <container_id>
```

### View running containers:
```bash
docker ps
```

### Remove the image:
```bash
docker rmi credit-default-app
```

## Project Structure

```
.
├── app.py              # Streamlit application
├── requirements.txt    # Python dependencies
├── Dockerfile         # Docker configuration
├── .dockerignore      # Docker ignore file
└── Models/            # Trained ML models
    ├── naive_bayes_best_model.pkl
    ├── Decision_tree_best_model.pkl
    └── best_svm_model_metadata.pkl
```

## Output

The application provides:
- **Prediction**: Whether the client will default (1) or pay (0)
- **Risk Assessment**: HIGH RISK or LOW RISK classification
- **Recommendations**: Actionable suggestions based on the prediction
- **Probability**: Default probability percentage (if available)

## Models

### Naive Bayes
- Probabilistic classifier based on Bayes' theorem
- Fast and efficient for classification tasks

### Decision Tree
- Tree-like model for decision making
- Easy to interpret and visualize

### SVM (Support Vector Machine)
- Finds optimal boundary between classes
- Effective in high-dimensional spaces

## Notes

- Ensure all model `.pkl` files are present in the `Models/` directory
- The application expects models trained on the specified feature set
- Input values are automatically validated and constrained to reasonable ranges
