# EVAT Voice Assistant – Data Science Review and Enhancement Plan

## Author
Harshit Gaur  
Data Science Team  

---

## Table of Contents
- [Overview](#overview)
- [Current System](#current-system)
- [System Architecture](#system-architecture)
- [Example Processing Flow](#example-processing-flow)
- [Observations](#observations)
- [Improvements That Can Be Implemented](#improvements-that-can-be-implemented)
  - [Intent Classification](#intent-classification)
  - [Entity Extraction](#entity-extraction)
  - [Direct Model Integration](#direct-model-integration)
  - [Response Improvement](#response-improvement)
  - [Logging](#logging)
  - [Expanding Query Types](#expanding-query-types)
- [Future Scope](#future-scope)
- [Conclusion](#conclusion)

---

## Overview

This document presents a technical review of the EVAT Voice Assistant and outlines practical improvements based on system evaluation and research.

The assistant allows users to interact with EVAT services using voice input. The system converts speech into text on the frontend and sends it to the backend for processing. The backend then interprets the query and returns a response.

The focus of this document is to:
- confirm what is currently working  
- explain the system structure  
- identify improvements that are easy to implement  

---

## Current System

The system is functioning end-to-end with a clear processing pipeline.

### Voice Input

Voice is captured on the frontend using browser-based speech recognition. The recognised text appears in real time, allowing users to confirm their input before submitting.

### API Communication

The recognised text is sent to the backend using a POST request:

```json
{
  "query": "Is the charger busy?"
}

The backend processes this request and returns a response, which is displayed to the user.

Backend Design

The backend only processes text and does not handle audio. This keeps the system simple and modular. The main responsibilities of the backend include:

understanding the query
extracting useful information
connecting to EVAT logic
generating a response
Model Integration

The assistant is designed to work with EVAT Data Science models. For example, the congestion prediction system uses a machine learning model to estimate charger usage and classify it as low, medium, or high.

System Architecture

The current system follows a simple pipeline:

User Voice Input
        ↓
Speech-to-Text (Frontend)
        ↓
Text Query
        ↓
POST /voice/query
        ↓
Intent Detection
        ↓
Entity Extraction
        ↓
EVAT Model (e.g., Congestion Prediction)
        ↓
Response Generation
        ↓
Frontend Display

This design separates concerns clearly:

frontend handles voice
backend handles logic
data science layer handles predictions
Example Processing Flow

A simplified backend logic can be represented as:

def process_query(query):
    intent = detect_intent(query)
    entities = extract_entities(query)

    if intent == "congestion":
        result = get_congestion(entities)
        return f"The charger is currently {result}."
    
    elif intent == "cost":
        result = calculate_cost(entities)
        return f"Estimated trip cost is {result}."
    
    return "Sorry, I could not understand your query."
Observations

The system is working correctly but has some limitations:

limited understanding of different query types
no structured extraction of key information
no support for follow-up queries
responses are basic and not very descriptive
Improvements That Can Be Implemented
Intent Classification

Add a simple intent detection layer using keyword matching or a lightweight model.

def detect_intent(query):
    q = query.lower()
    if "busy" in q or "charger" in q:
        return "congestion"
    if "cost" in q or "km" in q:
        return "cost"
    return "unknown"
Entity Extraction

Extract useful values such as location or distance.

import re

def extract_distance(query):
    match = re.search(r'(\d+)\s*km', query.lower())
    return int(match.group(1)) if match else None
Direct Model Integration

Connect the assistant directly to EVAT models such as congestion prediction.

def get_congestion(entities):
    station = entities.get("station")
    return "moderately busy"
Response Improvement

Improve how responses are generated so they sound clearer and more natural.

Example:

Instead of: "Busy"
Use: "The selected charging station is currently experiencing moderate congestion."
Logging

Add logging to track system usage and performance.

import logging

logging.basicConfig(filename="assistant.log", level=logging.INFO)

def log_query(query, intent):
    logging.info(f"{query} -> {intent}")
Expanding Query Types

Support more queries gradually, such as:

nearby charging stations
EV vs petrol comparison
general help queries
Future Scope

The assistant can later be extended to include:

multi-turn conversations
personalised recommendations
integration into EVAT dashboards
more advanced NLP models
Conclusion

The EVAT Voice Assistant is already functional and follows a solid design. The system successfully converts voice input into meaningful responses using a structured pipeline.

The next step is to improve how the system understands queries and connects to Data Science models. The suggested improvements are simple to implement and will significantly enhance the system without adding unnecessary complexity.