# InsightTalk.AI  
### **AI Agent for Voice-Driven Data Analysis**  

Welcome to **InsightTalk.AI**! This repository houses the code for a cutting-edge AI agent that enables seamless voice-driven interactions for data analysis. With InsightTalk.AI, users can leverage natural language commands to explore datasets, generate insights, and interact with analytics processes more intuitively than ever before.  

---

## **Features**  
- **Voice Interaction**: Speak naturally to query datasets and perform analytics.  
- **Data Insights**: Extract meaningful insights from raw data.  

---

## **Technologies Used**  
- **Python**: Core language for data processing and AI logic.  
- **Speech Recognition**: Whisper for converting voice input into text (Portuguese).  
- **LLM (Language Model)**: OpenAI's GPT models for query understanding and generating responses.  
- **Text-to-Speech (TTS)**: OpenAI's TTS API for converting the agent's response back to speech.  
- **Data Processing & Analysis**: Pandas for data manipulation and NumPy for numerical operations.  
- **Agent Framework**: LangChain for building conversational agents with tools like pandas dataframe agent.  
- **Audio Libraries**: SoundDevice and SoundFile for audio recording and processing.  
- **Environment Setup**: dotenv for environment variable management.

---

## **Getting Started**  

### **Prerequisites**  
- Python 3.8+  
- Virtual Environment Tool (e.g., `venv` or `conda`)  
- Internet connection for API-based voice recognition.  

### **Installation**

#### 1. Clone the repository:  
```bash
git clone https://github.com/Victor-Amarante/ai-agent-data-analysis.git
cd ai-agent-data-analysis
```

#### 2. Set up a virtual environment (optional, but recommended):
```bash
python3 -m venv venv
source venv/bin/activate
venv\Scripts\activate
```

#### 3. Install dependencies:
```bash
pip install -r requirements.txt
```

#### 4. Run the main script:
```bash
cd src/
python3 main.py
```