FROM python:3.14-slim

# JVM + Kaitai Struct compiler (for compiling our own candidate .ksy in later phases)
RUN apt-get update && apt-get install -y --no-install-recommends \
      default-jre-headless wget ca-certificates && rm -rf /var/lib/apt/lists/*
ARG KSC_VERSION=0.11
RUN wget -qO /tmp/ksc.deb \
      https://github.com/kaitai-io/kaitai_struct_compiler/releases/download/${KSC_VERSION}/kaitai-struct-compiler_${KSC_VERSION}_all.deb \
 && apt-get update && apt-get install -y --no-install-recommends /tmp/ksc.deb \
 && rm /tmp/ksc.deb && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e ".[dev]"
CMD ["bash"]
