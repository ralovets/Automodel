# Training image plus the native core of the Claude-style tokenizer.
# Build from a directory holding the tokenizer_lab_native wheel:
#   docker build -f experiment/tokenizer.Dockerfile -t automodel:mini2-tok <wheel directory>
ARG BASE=automodel:mini2
FROM ${BASE}
COPY tokenizer_lab_native-*.whl /tmp/wheels/
RUN uv pip install --python /opt/venv/bin/python /tmp/wheels/*.whl && rm -rf /tmp/wheels
