#!/bin/bash
cd "$(dirname "$0")"
cp gui/main_window_WORKING_BACKUP.py gui/main_window.py
cp main_WORKING_BACKUP.py main.py
cp core/yolo_model_WORKING_BACKUP.py core/yolo_model.py
cp assets/best.pt.WORKING_BACKUP assets/best.pt
echo "Restored to working version!"
