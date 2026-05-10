# AgentMail - Agentic Email Classification System

## Overview

AgentMail is an interactive AI system that clusters, labels, and classifies emails using machine learning and LLM-assisted reasoning.

## Features

* Automatic dataset schema detection
* KMeans clustering or user-defined labels
* Gemini-powered cluster explanations
* Interactive relabeling and retraining
* New email classification

## Requirements

```bash
pip install pandas scikit-learn google-generativeai numpy
```

## Run

```bash
python3 main.py
```

## Example Dataset Path

```bash
./emaildata/email_classification_dataset.csv
```

## Workflow

1. Enter dataset path
2. Enter Gemini API key
3. Choose KMeans or custom labels
4. Review clusters
5. Relabel or retrain if desired
6. Classify new emails

## Files

* main.py → program entry point
* agent.py → AI logic
* utils.py → helpers
* results.txt → evaluation output

