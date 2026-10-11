# Uber Fare Prediction: EDA & Machine Learning

An end-to-end machine learning project exploring taxi-fare data,
performing exploratory data analysis (EDA), training regression models,
and serving fare predictions through a Streamlit app.

> **Dataset note:** Verify and cite the original dataset source before
> describing all records as Uber trips. The dataset schema resembles a
> taxi-fare prediction dataset.

## Project Structure

``` text
Uber-Analysis/
├── data/
│   └── uber.csv
├── models/
│   ├── best_model.pkl
│   └── model_comparison.csv
├── notebooks/
│   ├── 01_uber_eda.ipynb
│   └── 02_modeling.ipynb
├── .gitignore
├── app.py
├── README.md
└── requirements.txt
```

## Project Objectives

-   Explore the dataset and identify important patterns through EDA.
-   Inspect missing values, duplicates, unusual fares, and invalid
    geographic coordinates.
-   Engineer features such as estimated trip distance and pickup-time
    components.
-   Train and compare regression models for fare prediction.
-   Save the selected model and evaluate its performance using
    regression metrics.
-   Provide a simple Streamlit interface for making predictions.

## Dataset

The initial dataset contains **200,000 rows and 9 columns** according to
the EDA notebook.

Key fields include:

  Column                                    Description
  ----------------------------------------- --------------------------------
  `fare_amount`                             Recorded fare; target variable
  `pickup_datetime`                         Pickup date and time
  `pickup_longitude`, `pickup_latitude`     Pickup coordinates
  `dropoff_longitude`, `dropoff_latitude`   Drop-off coordinates
  `passenger_count`                         Recorded passenger count
  `key`                                     Record identifier or key field

**Dataset source:** Add the verified original dataset URL here.

## Exploratory Data Analysis

The EDA notebook (`notebooks/01_uber_eda.ipynb`) covers:

-   Dataset structure, data types, and summary statistics
-   Missing-value and duplicate checks
-   Fare distribution and outlier inspection
-   Passenger-count distribution
-   Pickup activity by hour
-   Fare relationships with passenger count and pickup time
-   Estimated trip distance and fare-versus-distance analysis
-   Correlation analysis
-   Geographic coordinate validation

### Data quality

The EDA found missing drop-off coordinate values and extreme coordinate
values. These should be investigated and validated **before**
calculating trip distance. Global latitude/longitude limits alone do not
ensure that a trip belongs to the intended service area.

Non-positive fares, implausible passenger counts, zero-distance trips,
and extreme fare values should be handled using documented rules rather
than removed blindly.

## Feature Engineering

The modeling notebook derives features such as:

-   `distance_km` --- estimated straight-line distance using the
    Haversine formula
-   `pickup_hour` --- hour of pickup
-   `pickup_day` --- day of month
-   `pickup_month` --- month
-   `pickup_dayofweek` --- day of week
-   `passenger_count` --- passenger count

The prediction target is `fare_amount`.

> Haversine distance is a straight-line estimate, not actual road
> distance. Invalid coordinates must be handled before this feature is
> calculated.

## Machine Learning

The modeling notebook compares:

1.  Linear Regression
2.  Random Forest Regressor
3.  Gradient Boosting Regressor

Models are evaluated using:

-   **MAE:** Mean Absolute Error
-   **RMSE:** Root Mean Squared Error
-   **R²:** Coefficient of Determination

The selected model is saved as `models/best_model.pkl`, and model
comparison results are saved as `models/model_comparison.csv`.

### Model results

Update this table using the latest values from
`models/model_comparison.csv`.

  Model                           MAE   RMSE    R²
  ----------------------------- ----- ------ -----
  Linear Regression               ---    ---   ---
  Random Forest Regressor         ---    ---   ---
  Gradient Boosting Regressor     ---    ---   ---

No performance values are included here until they are confirmed from an
actual run.

## Requirements

Install dependencies from the existing requirements file:

``` bash
pip install -r requirements.txt
```

It is recommended to use a virtual environment.

**Windows PowerShell**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## How to Run

Run commands from the `Uber-Analysis` project root.

### 1. Run the EDA notebook

``` bash
jupyter notebook
```

Open `notebooks/01_uber_eda.ipynb` and run its cells.

### 2. Train and evaluate models

Open `notebooks/02_modeling.ipynb` and run its cells from top to bottom.
Ensure the notebook reads the CSV from `data/uber.csv` and saves outputs
under `models/`.

### 3. Launch the Streamlit app

``` bash
streamlit run app.py
```

The app expects the trained model at `models/best_model.pkl`. If the
file is missing, run the modeling notebook first. Use paths based on the
project root or paths resolved relative to `__file__` so the app works
consistently.

## Limitations

-   Predictions reflect patterns in the available dataset and selected
    features.
-   Straight-line distance does not capture road routes, traffic, tolls,
    or route choices.
-   Poor-quality coordinates and outliers can distort the analysis and
    model.
-   The model produces estimates, not official fare quotes.
-   Consider a time-based holdout evaluation to measure performance on
    later trips.

## Future Improvements

-   Add automated data validation and unit tests.
-   Build a reusable preprocessing pipeline shared between training and
    inference.
-   Add cross-validation and hyperparameter tuning.
-   Track experiments and version models.
-   Add CI checks and a repeatable training workflow.
-   Introduce continuous training (CT) with explicit data-quality
    checks, evaluation thresholds, and model-promotion rules.
-   Deploy the app and monitor prediction quality and data drift.

## Author

**Depesh Kumar**
*AI Engineer Intern*

-   GitHub: https://github.com/dipesh50061-blip
-   Portfolio: https://sites.google.com/view/depesh-ai-journey
-   LinkedIn: https://www.linkedin.com/in/depesh-kumar-918626399/

## License

Add a license before distributing this project publicly. Follow the
dataset provider's usage terms as well.
