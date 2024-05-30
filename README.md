# GGUF_GUI: An easy way to convert your safetensors to GGUF

Easy installation:

```shell
./run.sh
```

This should pull the repos and install the requirements.
You will need llama.cpp build using make or cmake. The script above will try to do that.

After the initial run you can just run:

```shell
streamlit run main.py
```

You can do this with CUDA as well.

To use the huggingface downloader you have to enter in the repo id:
for example: `username_or_org/repo_name` or `lysandre/test-model`.

![alt text](main.png "Main")

## Docker

You can also build the apps as a container image:

```shell
cp Dockerfile.cpu Dockerfile # or Dockerfile.cuda
docker build -t gguf_gui .
docker run -v /path/to/your/models:/app/models -p 8501:8501 gguf_gui
```
