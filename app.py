import sys, subprocess
from pathlib import Path

def run_back():
    # просто для проверки, можно показать help оригинального InferSim
    subprocess.run([sys.executable, "back/main.py", "--help"], cwd=Path(__file__).parent)

def run_front():
    subprocess.run(["streamlit", "run", "main.py"], cwd=Path(__file__).parent / "front")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python app.py [back|front]")
        sys.exit(1)
    cmd = sys.argv[1].lower()
    if cmd == "back":
        run_back()
    elif cmd == "front":
        run_front()
    else:
        print("Неизвестная команда. Используйте back или front.")