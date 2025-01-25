install:
	pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	pip3 install -r requirements.txt

run:
	py main.py
