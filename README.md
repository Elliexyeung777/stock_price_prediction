# stock_price_prediction
# Stock Prediction Project

This project aims to predict stock prices using machine learning and deep learning techniques.

## Project Structure

- `ml_pipeline/`
  - `train.py`: Main training script
  - `utils.py`: Contains various helper functions
- `lib/`
  - `stock_prediction_notebook.ipynb`: Jupyter notebook for exploratory data analysis and model development
- `engine.py`: Contains the `StockPredictionEngine` class for stock prediction

## Features

1. Fetch stock data from Yahoo Finance
2. Data preprocessing and feature engineering
3. Stock price prediction using different machine learning models:
   - Simple RNN (Recurrent Neural Network)
   - LSTM (Long Short-Term Memory)
   - Dense Neural Network
4. Model evaluation and visualization

## Usage

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the training script:
   ```
   python ml_pipeline/train.py
   ```

3. Use the `StockPredictionEngine`:
   ```python
   from engine import StockPredictionEngine
   
   engine = StockPredictionEngine()
   # Use the engine for predictions
   ```

4. Explore the Jupyter notebook for more in-depth analysis.

## Key Components

- `train.py`: Contains the complete pipeline for data loading, preprocessing, model training, and evaluation.
- `utils.py`: Provides various helper functions for data retrieval, preprocessing, and visualization.
- `engine.py`: Implements the `StockPredictionEngine` class, encapsulating the prediction process.
- `stock_prediction_notebook.ipynb`: Used for interactive data analysis and model development.

## Notes

- Ensure you have a stable internet connection to fetch data from Yahoo Finance.
- Model performance may vary depending on the stock and time range.
- This project is for educational purposes only and should not be considered financial advice.

## Contributing

Feel free to open issues, suggest improvements, or contribute code directly. Please follow the standard GitHub flow: fork, modify, and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
