# 🧠 Interactive Neural Network Learning Demo

A web-based, interactive demo built with **TensorFlow/Keras** and **Streamlit**.
Change the dataset, layers, neurons, activation function, learning rate and epochs,
press **Train**, and watch a real neural network learn.

**Live demo:** _paste your Streamlit URL here_

## Features
- Datasets: Moons, Circles, XOR, Two clusters (adjustable noise and size)
- Controls: hidden layers, neurons, activation (ReLU/Tanh/Sigmoid/Linear), learning rate, epochs, seed
- Visuals: network architecture (weights coloured by sign), decision boundary with an epoch replay slider,
  loss/accuracy curves, live prediction for a point you choose

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Use Python 3.10–3.12 (TensorFlow does not support newer versions reliably).

## Project structure
```
app.py                 # the whole application
requirements.txt       # dependencies
.streamlit/config.toml # theme
```
