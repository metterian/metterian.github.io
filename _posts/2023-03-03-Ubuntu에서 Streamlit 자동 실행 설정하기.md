---
layout: post
title: "Ubuntu에서 Streamlit 자동 실행 설정하기"
author: "metterian"
tags: "개발"
---

Streamlit은 파이썬으로 웹 애플리케이션을 쉽게 만들 수 있도록 도와주는 라이브러리입니다. Streamlit을 사용하여 만든 애플리케이션을 서버에서 실행할 때, 일반적으로 tmux나 screen과 같은 세션 관리 도구를 사용하여 백그라운드에서 실행합니다. 하지만 이 방법은 서버가 재부팅될 때마다 Streamlit을 다시 실행해야한다는 불편함이 있습니다.

이번에는 서버가 자동으로 부팅될 때 Streamlit을 자동으로 실행하도록 설정하는 방법을 알아보겠습니다.

먼저, systemd 서비스 파일을 만들어야 합니다. systemd는 리눅스에서 프로세스 관리를 담당하는 시스템 도구입니다. 서비스 파일은 /etc/systemd/system/ 디렉토리에 저장됩니다.

아래는 서비스 파일의 예시입니다.

```
[demo.service]
[Unit]
Description=CDL Demo
After=network.target

[Service]
Type=simple
User=persuade
WorkingDirectory=/home/persuade/HT.V0.3.23/
ExecStart=/bin/bash start_demo.sh
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

위의 서비스 파일에서는 User를 현재 사용자 이름으로 변경하고, WorkingDirectory를 Streamlit 애플리케이션 파일이 있는 디렉토리 경로로 변경해야 합니다. 또한, ExecStart에서는 start_demo.sh 스크립트를 실행합니다.

다음으로 start_demo.sh 스크립트 파일을 만들어야 합니다. 이 스크립트 파일에서는 Streamlit을 실행하기 전에 conda 환경을 활성화하고, Streamlit 애플리케이션 파일을 실행합니다.

```bash
#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cdl
streamlit run [demo.py](http://demo.py/) --server.port=9074
```

이제 `sudo systemctl start demo.service` 명령어를 사용하여 서비스를 시작하고, `sudo systemctl enable demo.service` 명령어를 사용하여 서비스가 부팅시 자동으로 실행되도록 설정합니다.

이제 서버가 자동으로 부팅될 때마다 Streamlit 애플리케이션도 함께 자동으로 실행됩니다.
