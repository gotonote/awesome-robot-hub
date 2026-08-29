# Service Robots

## Contents

- [1. Service Robot Overview](#1-service-robot-overview)
- [2. Indoor Navigation](#2-indoor-navigation)
- [3. Human-Robot Interaction](#3-human-robot-interaction)
- [4. Application Scenarios](#4-application-scenarios)

---

## 1. Service Robot Overview

### 1.1 Application Domains

- Reception / tour guiding
- Delivery
- Cleaning
- Companionship
- Medical assistance

### 1.2 Key Technologies

```
Perception → Understanding → Decision → Execution
```

---

## 2. Indoor Navigation

### 2.1 SLAM

```python
# Cartographer configuration
from cartographer_ros import Cartographer

cartographer = Cartographer()
cartographer.configure({
    'submaps': 3,
    'range_data_providers': ['laser', 'imu']
})
```

---

## 3. Human-Robot Interaction

### 3.1 Speech Interaction

```python
# Speech recognition and dialogue
from speech_recognition import Recognizer
from chatbot import DialogueSystem

recognizer = Recognizer()
dialogue = DialogueSystem()

def process_speech():
    audio = recognizer.listen()
    text = recognizer.recognize_google(audio)
    response = dialogue.generate(text)
    speak(response)
```

---

## 4. Application Scenarios

### 4.1 Restaurant Service

```
Workflow:
1. Greet guests
2. Guide to seats
3. Take orders
4. Serve food
5. Clear tables
6. See guests out
```

---

*This chapter is continuously updated...*
