@echo off
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat
echo Upgrading pip...
python -m pip install --upgrade pip
echo Installing requirements...
pip install -r requirements.txt
echo Setup complete. You can now run the tool using:
echo venv\Scripts\activate.bat
echo python main.py inputs\sample.mp3
