
![Screenshot 2024-08-05 184023](https://github.com/user-attachments/assets/3c3b90d3-e99e-4387-af1d-21335f738147)



![Screenshot 2024-08-05 184034](https://github.com/user-attachments/assets/268321df-64c4-4ab8-8c27-98ad44a499ec)


## About Random Forest Hyperparameter Tuning Dashboard

This project provides an interactive dashboard for tuning the hyperparameters of a Random Forest classifier using Streamlit. The dashboard allows users to easily adjust various hyperparameters and observe their impact on model performance through visualizations and metrics.

### Features

- **Interactive Hyperparameter Tuning**: Adjust key hyperparameters such as the number of estimators, maximum depth, minimum samples split, and minimum samples leaf using intuitive sliders.
- **Real-time Performance Metrics**: View classification reports and accuracy scores instantly as hyperparameters are adjusted.
- **Visualization of Results**: Heatmaps and confusion matrices provide a clear visual representation of model performance across different hyperparameter settings.
- **Ease of Use**: Built with Streamlit, the dashboard is easy to set up and use, requiring minimal configuration.

### Getting Started

To get started with the Random Forest Hyperparameter Tuning Dashboard, follow these steps:

1. **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/random-forest-tuning-dashboard.git
    cd random-forest-tuning-dashboard
    ```

2. **Install Dependencies**
    Ensure you have Python installed, then install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Dashboard**
    Launch the Streamlit app:
    ```bash
    streamlit run random_forest_tuning.py
    ```

### How It Works

1. **Data Loading**: The dashboard uses the Iris dataset for demonstration purposes. You can easily modify the code to load and use your dataset.
2. **Hyperparameter Tuning**: The `GridSearchCV` from `scikit-learn` is used to perform an exhaustive search over specified parameter values for a Random Forest classifier.
3. **Visualization**: `matplotlib` and `seaborn` are used to generate heatmaps and confusion matrices to visualize the results of the hyperparameter tuning.

### Customization

You can customize the dashboard to fit your specific needs by:
- Modifying the hyperparameter ranges and values in the `random_forest_tuning.py` script.
- Replacing the Iris dataset with your own dataset.
- Adding more visualization options or metrics to the dashboard.

### Contributions

Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

