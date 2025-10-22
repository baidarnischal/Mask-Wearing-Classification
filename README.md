# Mask Wearing Classification

A deep learning project to classify images into mask wearing categories: `mask_weared_incorrect`, `with_mask`, and `without_mask`. This project demonstrates end to end deep learning workflow including data loading, model training, hyperparameter tuning, saving/loading models, and deployment using Docker.

---

## Project Structure & File Descriptions

- **`api/`** – FastAPI backend  
  - `main.py` – Main FastAPI application with endpoints and model loading  
  - `requirements.txt` – Python dependencies for the API  

- **`frontend/`** – Frontend static files  
  - `index.html` – Main webpage  
  - `script.js` – JavaScript for frontend interactivity  
  - `style.css` – Styles for frontend  

- **`kaggle_notebooks/`** – Kaggle notebooks for training and tuning  
  - `01_hyperparameter_tuning_and_clearml_connecting_using_kaggle.ipynb` – Hyperparameter tuning notebook  
  - `02_hyperparams_tuned_model_is_trained_again_on_250_epochs.ipynb` – Notebook to train the final model (tuned model) with 250 epochs including early stopping  

- **`models/`** – Saved trained models (ignored in Git)

- **`training/`** – Training notebooks and dataset  
  - `Dataset/` – Training dataset (ignored in Docker and Git)  
  - `01_training_without_earlystopping.ipynb` – Notebook to train model without early stopping  
  - `02_loading_earlystopping_model.ipynb` – Load and evaluate early stopping model that was trained in kaggle with GPU  

- **`myenv/`** – Python virtual environment (ignored in Git/Docker)  

- **`.cz.yaml`** – Commitizen configuration file  (ignored in Docker)

- **`.gitignore`** – Files and folders ignored by Git  (ignored in Docker)

- **`Dockerfile`** – Dockerfile to containerize the project  

- **`.dockerignore`** – Files and folders ignored during Docker image build  

---
## Features
- Load dataset efficiently using `tf.data.Dataset`, caching, shuffling, and prefetching  
- Data augmentation for better generalization  
- Build CNN models with TensorFlow/Keras  
- Train models in Kaggle notebooks  
- Save trained models (`.keras`) and pickle files  
- Plot training metrics and visualize predictions  
- Hyperparameter tuning using **Keras Tuner** with random search  
- Implement dropout, early stopping, and experiment with different optimizers (`adam`, `rmsprop`, `SGD`)  
- Dockerized FastAPI backend with frontend support  
---

## Clone the repository

```bash
git clone https://github.com/bainash10/Mask-Wearing-Classification
cd mask_wearing_classification
```

## Run the project using docker 
### 1. Pull the docker image
```bash
docker pull nischal001/mask_wearing_classification
```

### 2. Run the docker container
```bash
docker run -p 8000:8000 nischal001/mask_wearing_classification
```