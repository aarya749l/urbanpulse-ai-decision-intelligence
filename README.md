# UrbanPulse 🚦

## AI-Powered Decision Intelligence for Smart Mobility

UrbanPulse is an intelligent urban mobility assistant built for the **Google Gen AI Academy APAC Edition – AI for Better Living and Smarter Communities Challenge**.

The platform leverages **Gemini 2.5 Flash on Vertex AI** to provide predictive analytics, conversational insights, and actionable recommendations for citizens, commuters, and city planners.

---

## Problem Statement

Modern communities generate vast amounts of transportation and public infrastructure data. However, transforming this information into meaningful decisions remains difficult.

UrbanPulse addresses this challenge by combining Generative AI, predictive analytics, and intelligent automation to support better urban mobility decisions.

---

## Features

### Traffic Congestion Forecasting

Predict future congestion levels and recommend optimal travel times.

### Public Transit Ridership Prediction

Forecast passenger demand to improve transit planning.

### Parking Availability Intelligence

Estimate parking occupancy and suggest better alternatives.

### Mobility Anomaly Detection

Detect unusual traffic patterns, events, and disruptions.

### Conversational AI Interface

Users interact naturally using everyday language.

### Interactive Forecast Visualizations

Generated predictions are automatically displayed as charts.

---

## Technology Stack

* Gemini 2.5 Flash
* Vertex AI
* Google Cloud Platform
* Streamlit
* Python
* Pandas
* Docker
* Cloud Run

---

## Google Cloud Services Used

* Vertex AI
* Gemini API
* Cloud Run
* Cloud Build
* Artifact Registry

---

## System Architecture

User → Streamlit UI → Gemini 2.5 Flash → Function Calling → Predictive Analytics Tools → Insights + Charts

---

## Running Locally

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"

streamlit run app.py --server.port 8080 --server.address 0.0.0.0
```

---

## Future Work

* BigQuery integration
* Real-time traffic sensor data
* IoT-based parking systems
* Smart city dashboards using Looker
* Disaster response analytics
* Citizen feedback analysis using RAG

---

## Team

Team Name: UrbanPulse

Built for Google Gen AI Academy APAC Edition.
