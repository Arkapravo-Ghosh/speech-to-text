# Speech-To-Text API

## Description

This is a Simple API built using Python and FastAPI that converts speech to text using OpenAI's Whisper v2 Model from HuggingFace Transformers.

## Installation

### Clone the repository and navigate to the directory

```bash
git clone https://github.com/Arkapravo-Ghosh/speech-to-text.git
```

```bash
cd speech-to-text
```

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate the virtual environment

<details>
  <summary>Windows</summary>

```pwsh
.\.venv\Scripts\activate.ps1
```

</details>

<details>
  <summary>GNU/Linux or macOS</summary>

```bash
source .venv/bin/activate
```

</details>

### Upgrade pip and install the dependencies

```bash
pip install -U pip
```

```bash
pip install -r requirements.txt
```

### Run the server

```bash
python main.py
```

## Usage

## Test with given sample audio file

```bash
curl -X POST -F "file=@./sample.m4a" "http://localhost:5000/transcribe"
```

- `/transcribe` (POST) - Transcribes the audio file sent in the request body and returns the transcript as a JSON response. Accepts `audio/wav`, `audio/mp3` or `audio/m4a` content types.
