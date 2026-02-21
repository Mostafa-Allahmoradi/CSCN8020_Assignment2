# Reinforcement Learning Programming (CSCN8020) - Assignment 2

## 👥 Author

Mostafa Allahmoradi - 9087818

## 📌 Assignment Overview

This assignment demonstrates the implementation of Q-Learning and Deep Q-Learning (DQN) within the 'Taxi-v3' OpenAI Gym environment.

For a detailed overview of the project requirements, please refer to CSCN8020_Assignment2.pdf located in the root directory.

## 🎯 How to Run:

1. **Clone this repository:**

   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```
2. **Create a Virtual Environment**
* Windows:
    ```bash
   python -m venv .venv
   ```

* macOS / Linux:
```bash
   python3 -m venv venv
   ```

3. **Activate the Virtual Environment**
* On Windows (Command Prompt):
    ```bash
   .venv\Scripts\Activate
   ```

* On macOS / Linux:
    ```bash
   source venv/bin/activate
   ```

4. **Install Required Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Enjoy the fun  game**

    ```bash
   python yellow_drift.py --episodes 5000 --epsilon-decay --log-file training_log.txt
   ```
