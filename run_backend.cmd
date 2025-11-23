@echo off
cd /d %~dp0
set LOCAL_LLM_PATH=google/flan-t5-small
set LOCAL_LLM_MAX_NEW_TOKENS=120
set LOCAL_LLM_TEMPERATURE=0.0
set LOCAL_LLM_NUM_BEAMS=4

REM Start server (keeps console open)
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
pause
