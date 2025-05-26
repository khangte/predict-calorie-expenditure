#!/bin/bash

# 커밋 메시지 입력 받기
read -p "커밋 메시지를 입력하세요: " user_message

# 현재 시간 (KST)
current_time=$(TZ=Asia/Seoul date "+%Y-%m-%d %H:%M:%S")

# 운영체제 및 버전 감지
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if [ -f /etc/os-release ]; then
        # Ubuntu 또는 다른 리눅스 배포판 이름 추출
        . /etc/os-release
        os_type="$NAME $VERSION"
    else
        os_type="Linux (Unknown Version)"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    os_type="macOS $(sw_vers -productVersion)"
elif [[ "$OSTYPE" == "cygwin" || "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows Git Bash 또는 WSL일 가능성 있음
    if command -v systeminfo &> /dev/null; then
        win_version=$(systeminfo | grep -E "^OS Name|^OS Version" | tr '\n' ' ' | sed 's/^.*OS Name: //; s/ OS Version:/ \(/; s/$/\)/')
        os_type="Windows $win_version"
    else
        os_type="Windows (Version Unknown)"
    fi
else
    os_type="Unknown OS"
fi

# 커밋 메시지 구성
commit_message="$user_message | $current_time (KST) | $os_type"

# Git 명령어 실행
git add .
git commit -m "$commit_message"
git push