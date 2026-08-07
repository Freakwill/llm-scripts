# 直接在当前终端执行
killollama() {
    local pid=$(lsof -ti:11434)
    if [ -n "$pid" ]; then
        kill -9 $pid
        echo "Ollama server on port 11434 stopped"
    else
        echo "No Ollama server running on port 11434"
    fi
}
killollama