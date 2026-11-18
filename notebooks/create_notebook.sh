#!/bin/bash
#SBATCH --job-name=KalmanDEG
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-16:00:00
#SBATCH --mem=24G
#SBATCH --output=logs_jupyter/jupyter_%j.log
#SBATCH --mail-user=stevenaragon314@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

trap 'kill $ARANGO_PID 2>/dev/null' EXIT

mkdir -p logs/logsAdultDB logs/logs_arango

mkdir -p logs_jupyter

if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    module load anaconda3 2>/dev/null || module load conda 2>/dev/null
fi

eval "$(conda shell.bash hook)"

conda activate llm_rag

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

IP_HOST=$(hostname -I | cut -d' ' -f1)
PORT=$(shuf -i 8000-9999 -n 1)
USER_HPC=$SLURM_JOB_USER
LOGIN_NODE=172.16.24.2
ACCESS_TOKEN='mitoken123'

echo "================================================================="
echo "Recordar que se debe de tener instalado Jupyter Lab en su computadora local y también el entorno/ambiente con los contenedores corriendo."
echo "¡Servidor Jupyter Lab solicitado e iniciado con éxito!"
echo "Asignado en el nodo de cómputo: $IP_HOST"
echo "Puerto de redirección en el clúster: $PORT"
echo "Tiempo máximo asignado: 04:00:00"
echo "================================================================="
echo "Para conectarte desde tu computadora local, ejecuta:"
echo ""
echo "ssh -L ${PORT}:${IP_HOST}:${PORT} ${USER_HPC}@${LOGIN_NODE}"
echo ""
echo "En caso de querer abrirlo en VSCODE deben de hacer:"
echo "1. Abre tu archivo .ipynb en VSCode."
echo "2. En la esquina superior derecha del notebook, haz clic en Select Kernel (o Seleccionar Núcleo)." 
echo "3. Selecciona Existing Jupyter Server... (o Servidor de Jupyter existente...)."
echo "4. Pega la URL: http://localhost:$PORT/lab?token=$ACCESS_TOKEN"
echo "5. En este paso va a aparecer 'localhost', solo hay que dar enter."
echo "6. Selecciona el entorno 'venv'"
echo ""
echo "Se tiene una segunda opción, la cual es realizar conexión vía Web, lo cual es más fácil"
echo "1. Abre tu navegador"
echo "2. Pega la URL: http://localhost:$PORT/lab?token=$ACCESS_TOKEN"
echo "3. Para finalizar, haz clic en el botón Connect (ConectarPresiona Enter, asigna un nombre al servidor si te lo pide, ¡y listo! Selecciona tu entorno 'venv'."
echo ""
echo ""
echo "Ejecuten este comando en Jupyter:"
echo "import socket"
echo "print(socket.gethostname())"
echo ""
echo "Esto debe de indicar el nombre del nodo, no puede decir 'accessnode'."
echo ""
echo "Si dice 'accessnode', entonces deben de cerrar la sesión y volver a iniciar, ya que está mal configurada la sesión."
echo ""
echo "================================================================="
echo ""


srun --exact -n 1 -c $SLURM_CPUS_PER_TASK jupyter lab --no-browser --ip=0.0.0.0 --port=$PORT --ServerApp.token=$ACCESS_TOKEN --ServerApp.allow_origin='*' --ServerApp.allow_remote_access=True

wait $ARANGO_PID 2>/dev/null