# Medium Articles Classifier

This project demonstrates a simple machine learning pipeline to classify Medium articles based on their relevance to "Data Science." Using a dataset of Medium articles, the project implements a Logistic Regression model trained on the full text of the articles, with a focus on extracting insights and building a predictive tool.

## Features

- **Data Loading**: Automatically fetches a dataset of Medium articles from the Hugging Face Hub.
- **Data Preprocessing**: Combines the title and article text to create a comprehensive feature set for training.
- **Binary Classification**: Identifies whether an article is related to Data Science or not based on its tags and content.
- **N-gram Analysis**: Highlights the most impactful words or phrases contributing to predictions.
- **Interactive Prediction**: Includes a function to predict the classification of any given text.

## Installation

### Prerequisites

- Python 3.8+

### Required Libraries

Install the required Python libraries using the following command:

```bash
pip install pandas scikit-learn huggingface_hub matplotlib
```

## Usage

### Clone the Repository

```bash
git clone https://github.com/python-srinu/Medium_Classifier_Data_Science.git
cd Medium_Classifier_Data_Science
```

### Interactive Predictions

Use the `predict_is_data_science` function to test the model with custom input text:

```python
sample_text = "This article discusses Python libraries for data analysis and machine learning."
print(predict_is_data_science(sample_text))
```

## Implementation Details

### Data Preprocessing

- **Tag-Based Labeling**: Articles are labeled as `True` if the tag "Data Science" is found, otherwise `False`.
- **Feature Engineering**: Combines the title and body text to generate a comprehensive `full_text` field.

### Model Training

- **Vectorization**: Utilizes the `CountVectorizer` from Scikit-learn to transform text data into numerical feature vectors.
- **Logistic Regression**: Trains a binary classifier on the vectorized text.

### Evaluation Metrics

- **Classification Report**:
  The model performance is evaluated using precision, recall, F1-score, and accuracy.
- **Confusion Matrix**:
  Visualizes true positives, true negatives, false positives, and false negatives.

### N-gram Insights

The most impactful n-grams (single words or phrases) contributing to predictions are extracted and sorted by their weights in the logistic regression model.

### Example Output

```text
Precision: 0.89
Recall: 0.88
F1-score: 0.88
Accuracy: 0.88
```

## Future Work

- Expand the dataset for improved generalization.
- Incorporate more advanced models like transformers for text classification.
- Build a web-based interface for user interaction with the classifier.

## Contributions

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.


For questions or opportunities, please reach out:

- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn Profile](#)
- **GitHub**: [Your GitHub Profile](#)
