# Fitcal Sleep ML Model Backend

This repository contains the backend implementation and act as MLs Endpoint that's call at core APIs through pub/sub as message broker for the Fitcal app.

## Project Description

The Fitcal backend is built using the Python programming language, and FastAPI framework. It provides API endpoints for user-related operations. The backend utilizes Firebase Auth Token for authentication, and Firestore for database NoSQL.

## Getting Started

To get started with the Fitcal MLs backend, follow these steps:

1. Clone the repository: `git clone https://github.com/CC-Bangkit-C242-PS153/Sleep-ML-APIs-Python.git`
2. Install the dependencies: `pip install -r requirements.txt`
3. Run the application: `python main.py`


## Dependencies

```
fastapi==0.115.5
google-cloud-firestore==2.19.0
grpcio==1.68.0
tensorflow==2.18.0
uvicorn==0.32.0
```

## Project Structure
```bash
├── README.md
├── .gitignore
├── firestoredb.py
├── main.py
├── requirements.txt
├── model
│   └── model.h5
```
