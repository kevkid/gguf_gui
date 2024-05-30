### GGUF_GUI: An easy way to convert your safetensors to GGUF

Easy installation:
```
bash run.sh
```
This should pull the repos and install the requirements.
You will need llama.cpp build using make or cmake. The script above will try to do that.

After the initial run you can just run:
```
streamlit run main.py
```
You can do this with CUDA as well

To use the huggingface downloader you have to enter in the repo id:
for example: `username_or_org/repo_name` or `lysandre/test-model`.


![alt text](main.png "Main")
