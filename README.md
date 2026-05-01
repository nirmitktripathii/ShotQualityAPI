# Shot Quality API

This repository contains a Flask API for serving the Expected Goals (xG) model.

## Features
- **Singleton Model Loading**: Efficient memory usage.
- **Dockerized**: Easy deployment.
- **Endpoints**: `/predict` (POST) and `/health` (GET).

## Usage
1. Build: `docker build -t shot-quality-api .`
2. Run: `docker run -p 5000:5000 shot-quality-api`
