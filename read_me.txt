#환경설정
python3.8 기준 필요 패키지 설치
cmd에 pip install -r requirements.txt 입력

#pyd파일 import하는법
:hiding된 rppg모듈 가져와 쓰기 위한 과정

1. import sys
   print(sys.path) 으로 현재 파이썬 경로 확인

2. 해당  파이썬 경로의 \Lib\site-packages 폴더에 get_rgb_functiondll.pyd과 functionsdll.pyd 파일을 포함시켜준다. 


예시) C:\Users\userl\anaconda3\Lib\site-packages

#SW 작동
code 폴더내의 main.py 돌리시면 됩니다!