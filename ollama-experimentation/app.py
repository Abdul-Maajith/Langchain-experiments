import ollama
stream  = ollama.chat(model='llama2', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)