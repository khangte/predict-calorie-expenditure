#!/bin/bash

#########################
# 파일 실행 방법
# . uv-activate.sh
#########################

# ✅ uv + Python 3.11 기반 가상환경 자동 설치 및 진입
echo "🔎 [INFO] uv 기반 가상환경 자동 설정 스크립트 실행 중..."

# 현재 쉘이 하위 셸인지 확인 (source로 실행된 경우 $0은 'bash', 아니면 파일명)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "
❗ 이 스크립트는 하위 셸에서 실행되고 있어 가상환경이 현재 셸에 적용되지 않습니다.
👉 반드시 아래와 같이 'source' 또는 '.' 명령으로 실행해야 합니다:

    source uv-activate.sh
    . uv-activate.sh <== 해당 명령 실행

⛔ 종료합니다.
"
  exit 1
fi

# 1. uv 설치 확인
if ! command -v uv &> /dev/null; then
    echo "⚠️ [WARN] uv가 설치되어 있지 않습니다. 설치를 시작합니다..."
    curl -Ls https://astral.sh/uv/install.sh | bash
    export PATH="$HOME/.cargo/bin:$PATH"  # uv 경로 추가 (필요시 .bashrc에 추가)
fi

# 2. .venv 존재 여부 확인
if [ ! -d ".venv" ]; then
    echo "📦 [SETUP] .venv이 없어 생성합니다 (Python 3.11 사용)..."
    uv venv --python 3.11 .venv
fi

# 3. 가상환경 진입
if [ -f ".venv/bin/activate" ]; then
    echo "✅ [ACTIVATE] .venv/bin/activate 진입"
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    echo "✅ [ACTIVATE] .venv/Scripts/activate 진입"
    source .venv/Scripts/activate
else
    echo "❌ [ERROR] activate 파일이 존재하지 않습니다."
    exit 1
fi
