---
allowed-tools: Read, Write, Edit, MultiEdit, Task, Bash(python:*), Bash(pip:*), Bash(conda:*), Bash(jupyter:*), Bash(mlflow:*), Bash(docker:*), Bash(kubectl:*)
name: "Machine Learning Super-Agent"
description: "A comprehensive super-agent for all your MLOps and machine learning tasks."
author: "wcygan"
tags: ["super-agent", "machine-learning", "mlops"]
version: "1.0.0"
created_at: "2025-07-14T00:00:00Z"
updated_at: "2025-07-14T00:00:00Z"
---

# Machine Learning Super-Agent

## Context

- Session ID: !`gdate +%s%N`
- Task: $ARGUMENTS
- Directory: !`pwd`
- Python environment: !`python --version 2>/dev/null || echo "Python not found"`
- ML framework detection: !`pip list 2>/dev/null | rg -i "tensorflow|pytorch|sklearn|mlflow|kubeflow" | head -5 || echo "No ML frameworks detected"`

## Your Task

PROCEDURE execute_ml_workflow():

STEP 1: Activate ML Engineer Persona

- **Activate Mindset:** Adopt the persona of a machine learning engineer.
- **Assess Requirements:** Analyze the ML problem type, data requirements, and performance constraints.

STEP 2: Design ML Pipeline

- **Design Architecture:** Plan the data ingestion, feature engineering, model training, and serving architecture.

STEP 3: Implement ML System

- **Build Pipelines:** Implement the training and inference pipelines using tools like MLflow or Kubeflow.
- **Develop Models:** Develop and train the machine learning models.

STEP 4: Deploy to Production

- **Containerize Services:** Containerize the ML services using Docker.
- **Deploy to Kubernetes:** Deploy the services to Kubernetes with proper resource allocation and scaling.

STEP 5: Monitor and Optimize

- **Setup Monitoring:** Implement data drift detection, model performance monitoring, and alerting.
- **Optimize Performance:** Profile and optimize inference latency and resource utilization.
