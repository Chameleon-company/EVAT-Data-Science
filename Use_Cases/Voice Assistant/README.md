# EVAT Voice Assistant – Trimester 3

**Author:** Mohtashim Misbah 
**Date:** Week 5  

---

## Table of Contents
- [1. Project Overview](#1-project-overview)  
- [2. Voice Interaction Architecture](#2-voice-interaction-architecture)  
- [3. In-Scope Functionality](#3-in-scope-functionality)  
- [4. Out-of-Scope Functionality](#4-out-of-scope-functionality)  
- [5. Deliverables](#5-deliverables)  
- [6. Justification for Scope](#6-justification-for-scope)  

---

## 1. Project Overview

In this trimester, I am working on building the EVAT Voice Assistant. The goal of this system is to help users access EV-related information more easily by asking simple, natural-language questions. Instead of navigating dashboards or searching through menus, users should be able to ask things like *“Is the charger busy?”* or *“How much would a 150 km EV trip cost?”* and get a quick answer.

To keep the scope realistic, I decided to make the backend fully **text-based**, while the frontend team can optionally add the voice input and output. This way, the backend stays focused on the core intelligence of the assistant: intent detection, entity extraction, connecting to the EVAT models, and generating responses. This structure also aligns with how real voice assistants like Alexa or Google Assistant handle their backend systems.

This project is meant to create a strong foundation that future trimesters can expand into a full multi-turn conversational assistant.

---

## 2. Voice Interaction Architecture

Even though this is called a “Voice Assistant,” the backend won’t process any audio. Instead, the workflow is simple and clean:

1. **User speaks into the mobile or web app.**  
   The app records the audio.

2. **Frontend converts the speech to text.**  
   They can use tools like:
   - Web Speech API  
   - Whisper  
   - Google Speech-to-Text  
   - Any STT the app team chooses  

3. **The backend receives the text version of the query.**  
   Example: “How busy is the Burwood charger?”

   Example request structure (placeholder):

```json
{
  "query": "<user text query>"
}
```

4. **I process the text on the backend:**
   - Detect the user’s intent  
   - Extract important information (like location or distance)  
   - Pass it to the correct EVAT use case  
   - Generate a clear and helpful response  

5. **The frontend displays the text response** or turns it back into audio if needed.

This architecture keeps things modular and avoids overcomplicating the backend. It also makes the system easier to test and scale later.

---

## 3. In-Scope Functionality

### 3.1 Intent Classification  
I will build a simple intent classifier that can recognise the main types of questions users might ask. For now, I am focusing on three core intents:

- **Congestion Status Query**  
  Example: “Is the Burwood charger busy right now?”
  
- **Trip Cost Comparison**  
  Example: “How much would a 150 km EV trip cost?”

- **Help / Unsupported Query**  
  Example: “What can you do?”

These cover the most important and realistic use cases for this trimester.

---

### 3.2 Entity Extraction  
The system will pull out key information from user queries, such as:

- Location names  
- Distances  
- Any values the models need to run  

This helps the assistant generate more accurate and personalised responses.

---

### 3.3 Integration with EVAT Use Cases  
I will integrate the Voice Assistant with two EVAT models that are already well-developed:

#### 1. Congestion Prediction  
- Provides charger busyness  
- Gives estimated wait times  
- Helps users plan ahead  

#### 2. EV vs Petrol Trip Cost Comparison  
- Calculates EV trip cost  
- Calculates petrol cost  
- Helps users compare both options quickly  

These use cases are mature enough to work reliably with natural-language inputs.

---

### 3.4 Response Generation  
The assistant will return short, easy-to-understand answers.  
Examples:

- “The Burwood charger is moderately busy with a 5-minute wait.”  
- “A 150 km trip would cost about $18 for EV and around $24 for petrol.”  

My goal is to keep the responses simple and practical.

---

### 3.5 Logging  
To meet HD-level requirements, I will also implement logging for:

- Inputs  
- Detected intents  
- Extracted entities  
- System responses  

This will help evaluate the performance in Week 8 and identify areas for improvement.

---

## 4. Out-of-Scope Functionality

### 4.1 Additional EVAT Use Cases  
At this stage, I am **not** integrating with the other EVAT use cases because their models aren’t fully ready or they require more complex logic. Examples include:

- Environmental impact  
- Gamification  
- Charger rental  
- Usage insights  
- Weather-based routing  
- Reliability scoring  
- Site suitability  
- Demand forecasting  

These can be added in future trimesters.

---

### 4.2 Audio Processing  
The backend will not:
- Process audio  
- Handle speech recognition  
- Convert text to speech  

All of this is handled by the frontend team.

---

### 4.3 Multi-Turn Conversations  
The assistant will not remember previous queries or support follow-up questions.  
Each query is treated independently.

---

### 4.4 UI Development  
UI work is outside my scope.  
The frontend team will manage:
- Input boxes  
- Voice buttons  
- Display responses  

---

## 5. Deliverables

By Week 10, I plan to deliver:

- A working Voice Assistant backend  
- Intent classifier + entity extraction  
- Integrated handlers for congestion and cost comparison  
- A `/voice/query` production API  
- Logging + evaluation results  
- Full documentation (architecture, mapping, API details)  
- A short demo for the mentor/panel  

---

## 6. Justification for Scope

I chose this scope because it is achievable within the trimester, avoids unnecessary complexity, and focuses on delivering real value. This setup:

- Fits the 6-week timeline  
- Uses EVAT models that are already stable  
- Minimises dependencies on other teams  
- Keeps the backend clean and realistic  
- Sets up a strong base for future expansion  
- Meets HD expectations by including logging, evaluation, and clear architecture  

Overall, this scope gives the data team something functional and meaningful while keeping the workload manageable and focused.

