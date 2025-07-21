FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl libgl1 libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Set Arbeitsverzeichnis
WORKDIR /app

# Kopiere alle Projektdateien ins Image
COPY . /app

# Upgrade Pip
RUN pip3 install --upgrade pip==25.1.1

# Python alias setzen
RUN ln -s /usr/bin/python3 /usr/bin/python

# Pip aktualisieren
RUN pip install --upgrade pip


# Installiere alle benötigten Pakete mit festen Versionen
RUN pip3 install \
    accelerate==1.8.1 \
    certifi==2025.6.15 \
    charset-normalizer==3.4.2 \
    diffusers==0.34.0 \
    filelock==3.18.0 \
    fsspec==2025.5.1 \
    ftfy==6.1.1 \
    hf-xet==1.1.5 \
    huggingface-hub==0.33.2 \
    idna==3.10 \
    importlib_metadata==8.7.0 \
    Jinja2==3.1.4 \
    MarkupSafe==2.1.5 \
    mpmath==1.3.0 \
    networkx==3.3 \
    numpy==1.25.2 \
    packaging==25.0 \
    peft==0.16.0 \
    pillow==11.3.0 \
    psutil==7.0.0 \
    PyYAML==6.0.2 \
    regex==2024.11.6 \
    requests==2.32.4 \
    safetensors==0.5.3 \
    scipy==1.9.3 \
    setuptools==59.6.0 \
    sympy==1.13.3 \
    tokenizers==0.21.2 \
    torch==2.7.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 \
    tqdm==4.67.1 \
    transformers==4.53.1 \
    triton==3.3.1 \
    typing_extensions==4.14.1 \
    urllib3==2.5.0 \
    wcwidth==0.2.13 \
    wheel==0.37.1 \
    zipp==3.23.0

# Standardbefehl (anpassbar)
CMD ["python", "predict.py"]
