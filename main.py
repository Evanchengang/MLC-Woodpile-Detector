#!/usr/bin/env python3
import os
os.environ["PYTORCH_MPS_AVAILABLE"] = "0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

from gui.main_window import MainWindow

if __name__ == "__main__":
    app = MainWindow()
    app.run()
