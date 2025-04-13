#!/bin/bash

cd /home/rccuser/offline-transcription-diarization
nohup streamlit run ./app/main.py --server.port 8502
