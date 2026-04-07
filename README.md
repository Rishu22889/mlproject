# 🚢 End-to-End Machine Learning Project: Titanic Survival Prediction

## 📌 Overview

This project demonstrates a complete **end-to-end machine learning pipeline** built using the Titanic dataset. The goal is to predict whether a passenger survived or not based on various features.

Unlike basic ML notebooks, this project focuses on **production-level structure**, including modular pipelines, logging, exception handling, and deployment-ready design.

---

## ⚙️ Key Features

* 🔹 Modular pipeline-based architecture
* 🔹 Data ingestion, transformation, and model training pipelines
* 🔹 Custom logging and exception handling
* 🔹 Reusable utilities for scalability
* 🔹 Prediction pipeline for inference
* 🔹 Docker support for deployment
* 🔹 Web interface integration (UI added)

---

## 🧠 Project Architecture

```
mlproject/
│
├── src/ml_project/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_monitoring.py
│   │
│   ├── pipelines/
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   │
│   ├── logger.py
│   ├── exception.py
│   └── utils.py
│
├── app.py
├── setup.py
├── template.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🔄 Workflow

1. **Data Ingestion**

   * Load Titanic dataset
   * Handle raw data storage

2. **Data Transformation**

   * Feature engineering
   * Data preprocessing (handling missing values, encoding, scaling)

3. **Model Training**

   * Train ML model
   * Evaluate using performance metrics

4. **Model Monitoring**

   * Track model performance and logs

5. **Prediction Pipeline**

   * Accept input data
   * Return survival prediction

---

## 📊 Dataset

* Dataset Used: **Titanic Dataset**
* Task: Binary Classification (Survived / Not Survived)

---

## 🚀 Tech Stack

* Python
* Scikit-learn
* Pandas, NumPy
* Logging & Exception Handling
* Docker
* Flask / Streamlit (for UI)

---

## ▶️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/mlproject.git
cd mlproject
```

### 2. Create Virtual Environment

```bash
conda create -p venv python=3.10 -y
conda activate ./venv
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Training Pipeline

```bash
python src/ml_project/pipelines/training_pipeline.py
```

### 5. Run Application

```bash
python app.py
```

---

## 🐳 Docker Support

```bash
docker build -t mlproject .
docker run -p 5000:5000 mlproject
```

---

## 🎯 What I Learned

* Building **production-ready ML pipelines**
* Writing modular and scalable code
* Handling errors using custom exception classes
* Implementing logging for debugging
* Structuring ML projects like real-world systems
* Integrating ML models into applications

---

## 📌 Future Improvements

* Add CI/CD pipeline
* Deploy on cloud (AWS/GCP/Azure)
* Add model versioning (MLflow/DVC)
* Improve UI/UX

---

## 🤝 Contribution

Feel free to fork this repository and improve it!

---

## 📬 Contact

If you have any questions or suggestions, feel free to reach out.

---
