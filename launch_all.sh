#!/bin/bash
# sistema completo HRI/Vision — un solo comando

eval "$(/Users/agrm/miniconda3/bin/conda shell.bash hook)"
conda activate ros2_hri

rm -f /Users/agrm/miniconda3/envs/ros2_hri/lib/python3.9/site-packages/typing.py 2>/dev/null || true

echo "=============================="
echo "  HRI/Vision — RoBorregos"
echo "=============================="

# ollama en background si no esta corriendo
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "[+] iniciando ollama..."
    ollama serve &
    OLLAMA_PID=$!
    sleep 3
else
    echo "[+] ollama ya esta corriendo"
    OLLAMA_PID=""
fi

# mcp_server en background (FastAPI, no es nodo ROS)
PYTHONPATH="/Users/agrm/ros2_hri_ws/install/hri_rag/lib/python3.9/site-packages:/Users/agrm/ros2_hri_ws/install/hri_interfaces/lib/python3.9/site-packages" \
/Users/agrm/miniconda3/envs/ros2_hri/bin/python -m hri_rag.mcp_server &
MCP_PID=$!

cleanup() {
    kill $MCP_PID 2>/dev/null || true
    [ -n "$OLLAMA_PID" ] && kill $OLLAMA_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

sleep 2
echo "[+] mcp_server en http://localhost:8000"

# todos los nodos ROS en un solo proceso foreground
/Users/agrm/miniconda3/envs/ros2_hri/bin/python /Users/agrm/ros2_hri_ws/scripts/hri_system.py

cleanup
