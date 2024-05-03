FROM cnstark/pytorch:2.0.1-py3.10.11-ubuntu22.04

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
# required by cv2
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN mkdir -p /opt/app /inputs /outputs \
    && chown user:user /opt/app /inputs /outputs

USER user
WORKDIR /opt/app
ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user:user . .

RUN pip install -e . 