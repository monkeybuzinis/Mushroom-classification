# Mushroom Classification Project

A comprehensive machine learning project that predicts whether a mushroom is **edible** or **poisonous** using various classification algorithms. This project combines thorough exploratory data analysis with multiple machine learning models to achieve high-accuracy predictions.

## Overview

This project tackles a critical classification problem: accurately predicting mushroom edibility based on physical characteristics. Using the [Mushrooms dataset from Kaggle](https://www.kaggle.com/datasets/uciml/mushroom-classification), the project explores relationships between mushroom features and compares multiple machine learning algorithms to identify the most reliable classification model.

## Dataset

- **Source**: [Kaggle Mushroom Classification Dataset](https://www.kaggle.com/datasets/uciml/mushroom-classification)
- **Samples**: 8,143 mushroom observations
- **Features**: 23 categorical features describing physical characteristics
- **Target**: Binary classification (edible 'e' or poisonous 'p')
- **Data Quality**: Complete dataset with no missing values (except '?' in stalk-root)

### Features

The dataset includes 23 categorical features:
- **Cap characteristics**: shape, surface, color
- **Gill characteristics**: attachment, spacing, size, color
- **Stalk characteristics**: shape, root, surface (above/below ring), color (above/below ring)
- **Other features**: bruises, odor, veil type/color, ring number/type, spore print color, population, habitat

## Project Structure

```
Mushroom_Classification/
├── Mushroom_Classification__project_1.ipynb
├── mushrooms.csv
└── README.md
```

## Methodology

### Phase 1: Exploratory Data Analysis (EDA)
- Data quality assessment
- Feature distribution analysis
- Relationship exploration between features and target variable
- Statistical tests (chi-square tests for categorical relationships)
- Visualization using matplotlib, seaborn, and plotly
- Feature importance analysis using mutual information

### Phase 2: Machine Learning Models
The project implements and compares multiple classification algorithms:

1. **Naive Bayes** (GaussianNB)
2. **K-Nearest Neighbors** (KNN)
3. **Support Vector Machine** (SVM)
4. **Logistic Regression**
5. **Random Forest Classifier**
6. **XGBoost Classifier**

### Phase 3: Model Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix analysis
- ROC-AUC curves
- Model comparison and selection

### Phase 4: Model Interpretability
- **LIME (Local Interpretable Model-agnostic Explanations)** for explaining individual predictions
- Feature importance visualization
- Understanding model decision-making process

## Requirements

### Python Packages

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost lime scipy graphviz
```

### Key Libraries
- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`, `graphviz`
- **Machine Learning**: `scikit-learn`, `xgboost`
- **Model Interpretability**: `lime`
- **Statistical Analysis**: `scipy`

## Usage

1. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook Mushroom_Classification__project_1.ipynb
   ```

2. **Run cells sequentially** to:
   - Load and explore the dataset
   - Perform exploratory data analysis
   - Preprocess the data (label encoding, feature selection)
   - Train multiple classification models
   - Evaluate and compare model performance
   - Generate predictions and interpretability visualizations

## Key Findings

- **Data Quality**: The dataset is remarkably clean with zero missing values (except '?' in stalk-root)
- **Feature Analysis**: All features are categorical, requiring label encoding for ML algorithms
- **Uninformative Features**: 'veil-type' contains only one value across all samples
- **Model Performance**: Multiple algorithms achieve high accuracy on this well-structured dataset

## Results

The project includes comprehensive model comparison showing:
- Performance metrics for each algorithm
- Confusion matrices for classification visualization
- ROC curves for model evaluation
- Feature importance rankings
- LIME explanations for model interpretability

## Model Interpretability

The project uses **LIME** to explain individual predictions, helping understand:
- Which features contribute most to each classification
- How different features interact in the model's decision-making
- Why a specific mushroom is classified as edible or poisonous

## Visualizations

The notebook includes various visualizations:
- Feature distribution plots
- Correlation heatmaps
- Confusion matrices
- ROC curves
- Feature importance charts
- Interactive plots using Plotly

## Notes

- All features are categorical and encoded as single letters
- The dataset is well-balanced, making it ideal for classification
- Model performance may vary based on train-test split and hyperparameters
- LIME explanations provide insights into model behavior for individual predictions

## References

- [Kaggle Mushroom Classification Dataset](https://www.kaggle.com/datasets/uciml/mushroom-classification)
- [LIME Documentation](https://github.com/marcotcr/lime)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

## Author

**Thi Kim Khanh Le**

## License

This project uses the Mushroom Classification dataset from Kaggle. Please refer to the dataset's license for usage terms.

