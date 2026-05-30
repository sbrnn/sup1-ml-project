# sup1-ml-project

A KNN classifier that predicts iris species from flower measurements, automatically tuning k from 1 to 10.

## Requirements

- Python ≥ 3.8
- pandas ≥ 1.3
- scikit-learn ≥ 1.0

```bash
pip install pandas scikit-learn
```

## Usage

```bash
python knn_iris.py
```

To predict your own flower, edit this line in the script:

```python
new_flower = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]],
             columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
```

## How it works

Loads the Iris dataset, splits it 70/30, then trains a KNN model for each k from 1 to 10. The best-performing k is used to retrain a final model and predict the species of a new flower.
