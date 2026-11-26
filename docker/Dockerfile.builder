# Clean-room source builder for patched tensorfold-native (and optional libtfgrammar.so) on DGX Spark (aarch64).
# Applies engine-patches/series onto ashhart/TensorFold@02b324ae9767253d9911d753d5c835a5070eed1b (v1.3.5).
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

ARG TF_REPO=https://github.com/ashhart/TensorFold.git
ARG TF_COMMIT=02b324ae9767253d9911d753d5c835a5070eed1b
ARG ZIG_VERSION=0.17.0
ARG ZIG_SHA256=9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8
ARG XGRAMMAR_TAG=v0.2.8
ARG BUILD_GRAMMAR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git python3 xz-utils build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /tmp/zig.tar.xz "https://ziglang.org/download/${ZIG_VERSION}/zig-aarch64-linux-${ZIG_VERSION}.tar.xz" \
 && echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c - \
 && mkdir -p /opt/zig \
 && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 \
 && rm -f /tmp/zig.tar.xz
ENV PATH=/opt/zig:${PATH}

WORKDIR /src
RUN git clone --no-checkout "${TF_REPO}" tensorfold \
 && git -C tensorfold checkout "${TF_COMMIT}"

COPY engine-patches /src/engine-patches
RUN cd /src/tensorfold \
 && git config user.name "TensorFold Builder" \
 && git config user.email "builder@localhost" \
 && while IFS= read -r p; do \
      [ -n "$p" ] || continue; \
      git am "/src/engine-patches/$p"; \
    done < /src/engine-patches/series

RUN cd /src/tensorfold \
 && zig build -Dgpu=spark -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=ReleaseFast \
 && mkdir -p /out \
 && cp zig-out/native/bin/tensorfold-native /out/tensorfold-native

RUN if [ "${BUILD_GRAMMAR}" = "1" ]; then \
      git clone --branch "${XGRAMMAR_TAG}" --depth 1 --recurse-submodules https://github.com/mlc-ai/xgrammar /src/xgrammar && \
      cd /src/tensorfold && \
      zig build grammar -Dxgrammar=/src/xgrammar && \
      cp zig-out/native/lib/libtfgrammar.so /out/libtfgrammar.so; \
    fi
