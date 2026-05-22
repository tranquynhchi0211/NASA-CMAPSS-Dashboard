# NASA C-MAPSS Turbofan Engine Dashboard

A Streamlit dashboard for NASA C-MAPSS Turbofan Engine predictive maintenance analysis.

## Project structure

```text
cmapss_streamlit_dashboard/
  app.py
  train_lstm_fd001.py
  generate_predictions.py
  requirements.txt
  data/
  models/
  predictions/
  src/
    data_utils.py
```

## Data files

Put the following files in the `data/` folder:

```text
train_FD001.txt, test_FD001.txt, RUL_FD001.txt
train_FD002.txt, test_FD002.txt, RUL_FD002.txt
train_FD003.txt, test_FD003.txt, RUL_FD003.txt
train_FD004.txt, test_FD004.txt, RUL_FD004.txt
```

## Install

```bash
pip install -r requirements.txt
```

## Train and cache predictions

Train the FD001 LSTM model and save the model plus preprocessing pipeline:

```bash
python train_lstm_fd001.py
```

Generate prediction CSV files for dashboard speed:

```bash
python generate_predictions.py
```

This creates files such as:

```text
predictions/predictions_FD001.csv
predictions/predictions_FD002.csv
predictions/predictions_FD003.csv
predictions/predictions_FD004.csv
```

FD001 uses the saved LSTM model when available. Other datasets use the baseline prediction logic unless you add extra trained models.

## Run dashboard

```bash
streamlit run app.py
```
