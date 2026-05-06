import requests, json

# Define the local Ollama API endpoint
url = "http://localhost:11434/api/generate"

# Send a prompt to the Gemma 3 model
payload = {
    "model": "gemma:2b",
    "prompt": "Write a short story about space exploration."
}

# stream=True tells requests to read the response as a live data stream
response = requests.post(url, json=payload, stream=True)

# Ollama sends one JSON object per line as it generates text
for line in response.iter_lines():
    if line:
        data = json.loads(line.decode("utf-8"))
        # Each chunk has a "response" key containing part of the text
        if "response" in data:
            print(data["response"], end="", flush=True)
#This setup turns your computer into a local AI engine. You can integrate it with chatbots, coding assistants, or automation tools without using external APIs.