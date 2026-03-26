# EV Charging Station Dashboard with Congestion Prediction

This document provides instructions on how to set up and run the EV charging station dashboard, which includes a congestion prediction feature.

## Prerequisites

Before you begin, ensure you have Python installed on your system.

### Step 1: Create and Activate a Virtual Environment

It is recommended to use a virtual environment to manage the dependencies for this project.

1.  **Navigate to the `Dashboard` directory:**
    Open a terminal and navigate to the project's `Dashboard` folder:
    ```bash
    cd Dashboard
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    *   **On Windows:**
        ```bash
        venv\Scripts\activate
        ```
    *   **On macOS and Linux:**
        ```bash
        source venv/bin/activate
        ```

### Step 2: Install Dependencies

Once the virtual environment is activated, install the required packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## How to Run

The system consists of two main components that need to be run from the `Dashboard` directory in separate terminals.

1.  **Congestion Prediction API:** A FastAPI server that provides real-time congestion predictions.
2.  **Streamlit Dashboard:** An interactive web application for visualizing charging stations and their predicted congestion.

### Step 1: Start the Congestion Prediction API

1.  **Open a terminal** and navigate to the `Use_Cases/Congestion Prediction/Prediction` directory with your virtual environment activated.
    ```bash
    cd ../Use_Cases/Congestion\ Prediction/Prediction
    ```

2.  **Start the FastAPI server:**
    ```bash
    uvicorn model_api:app --reload --port 8000
    ```
    Keep this terminal running. You should see output indicating that the server is running.

### Step 2: Run the Streamlit Dashboard

1.  **Open a new terminal** and make sure you are in the `Dashboard` directory with your virtual environment activated.

2.  **Run the Streamlit application:**
    ```bash
    streamlit run dashboard.py
    ```
    This will automatically open a new tab in your web browser with the dashboard.

## Using the Dashboard

Once the dashboard is running, you can use the sidebar to:

1.  **Select a charging station** from the dropdown menu.
2.  **Choose a date** for the prediction.
3.  Click the **"Predict Congestion"** button.

The dashboard will then display the predicted congestion level for the selected station and update the map with a color-coded marker:

*   **Green:** Low congestion
*   **Yellow:** Medium congestion
*   **Red:** High congestion
*   **Blue:** Default color for unselected stations
