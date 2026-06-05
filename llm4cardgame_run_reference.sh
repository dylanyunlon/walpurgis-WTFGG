(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ (base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ cat  llm4cardgame_run.sh 
#!/bin/bash
# LLM4CardGame - Complete Training Pipeline
# Paper: Can Large Language Models Master Complex Card Games?
# Repository: https://github.com/THUDM/LLM4CardGame
# Dependencies: DouZero, DanZero, RLCard, LLaMA-Factory, OpenCompass

set -e

echo "=========================================="
echo "   LLM4CardGame Training Pipeline"
echo "   Card Games: Doudizhu, Mahjong, etc."
echo "=========================================="
echo ""

# ===========================================
# Configuration Section
# ===========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Main paths
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/LLM4CardGame}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"

# Training parameters (these can be overridden via environment variables)
# Note: Don't use positional parameters here since $1 is the command name
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"  # or meta-llama/Llama-3.1-8B-Instruct
TRAINING_MODE="${TRAINING_MODE:-sft}"  # sft, lora, full
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU for LoRA fine-tuning (sufficient for 7B models)
CUDA_DEVICE="${CUDA_DEVICE:-0}"  # Default GPU device ID (set to 2 for H100)
MASTER_PORT=${MASTER_PORT:-29500}

# Conda environment name
CONDA_ENV_NAME="llm4cardgame"

# 8 Card Games supported
GAMES=("doudizhu" "mahjong" "blackjack" "leduc_holdem" "limit_holdem" "uno" "gin_rummy" "bridge")

# ===========================================
# Utility Functions
# ===========================================

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

# ===========================================
# Check System Requirements
# ===========================================

check_system() {
    print_step "Checking System Requirements"
    
    # Check NVIDIA Driver
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader || {
        echo "Error: NVIDIA driver not found!"
        exit 1
    }
    
    # Check CUDA
    echo -e "\nCUDA Version:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "CUDA version not found"
    
    # Check GPU Info
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    
    # Check conda
    check_command conda || {
        echo "Error: Conda is required. Please install Miniconda or Anaconda first."
        exit 1
    }
    
    echo -e "\n✓ System check passed"
}

# ===========================================
# Environment Setup
# ===========================================

setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV_NAME"
    
    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        echo -n "Recreate environment? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            conda deactivate 2>/dev/null || true
            conda env remove -n ${CONDA_ENV_NAME} -y
        else
            echo "Using existing environment..."
            eval "$(conda shell.bash hook)"
            conda activate ${CONDA_ENV_NAME}
            return
        fi
    fi
    
    # Create environment
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV_NAME}
    
    # Install PyTorch
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch CUDA
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
"
    
    # Install base dependencies
    echo -e "\nInstalling base dependencies..."
    pip install transformers==4.51.0 datasets==3.1.0 accelerate==1.7.0 tokenizers==0.21.1
    pip install peft==0.13.2 trl==0.11.4 deepspeed==0.15.4
    pip install sentencepiece protobuf safetensors
    pip install numpy scipy pandas==2.0.0 matplotlib seaborn
    pip install tensorboard wandb rich typer
    
    # Install Flash Attention (optional, for better performance)
    echo -e "\nInstalling Flash Attention 2..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention installation failed (optional)"
    
    # Install DouZero for Doudizhu
    echo -e "\nInstalling DouZero (Doudizhu AI)..."
    pip install douzero || pip install git+https://github.com/kwai/DouZero.git
    
    # Install RLCard for other card games
    echo -e "\nInstalling RLCard (Multi-game framework)..."
    pip install rlcard
    
    # Install LLaMA-Factory for training
    echo -e "\nInstalling LLaMA-Factory..."
    pip install llamafactory || pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    
    # Install OpenCompass for evaluation
    echo -e "\nInstalling OpenCompass dependencies..."
    pip install opencompass || echo "⚠️  OpenCompass may need manual setup"
    
    echo -e "\n✓ Environment setup completed"
}

# ===========================================
# Clone Repository
# ===========================================

clone_repository() {
    print_step "Cloning LLM4CardGame Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        echo "Project directory already exists: $PROJECT_DIR"
        echo -n "Pull latest changes? (Y/n): "
        read -r response
        if [[ ! "$response" =~ ^[Nn]$ ]]; then
            cd $PROJECT_DIR
            git pull
            cd $SCRIPT_DIR
        fi
    else
        git clone https://github.com/THUDM/LLM4CardGame.git $PROJECT_DIR
    fi
    
    # Create necessary directories
    mkdir -p $DATA_DIR
    mkdir -p $OUTPUT_DIR
    mkdir -p $MODEL_DIR
    mkdir -p $PROJECT_DIR/logs
    
    echo "✓ Repository ready at: $PROJECT_DIR"
}

# ===========================================
# Generate Game Data
# ===========================================

generate_game_data() {
    print_step "Generating Game Data (gen_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if data already exists
    if [ -d "$DATA_DIR/raw" ] && [ "$(ls -A $DATA_DIR/raw 2>/dev/null)" ]; then
        echo "Game data already exists in $DATA_DIR/raw"
        echo -n "Regenerate data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data generation..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    echo "This step generates interaction data from teacher models for 8 card games:"
    echo "  - Doudizhu (DouZero)"
    echo "  - Mahjong, Blackjack, Leduc Holdem, Limit Holdem (RLCard)"
    echo "  - UNO, Gin Rummy, Bridge (RLCard)"
    echo ""
    
    # Create data generation script
    

    # Run data generation
    echo "Starting data generation..."
    python gen_data_impl.py \
        --output_dir $DATA_DIR/raw \
        --num_episodes ${NUM_EPISODES:-10000} \
        --games doudizhu mahjong blackjack leduc-holdem limit-holdem uno gin-rummy
    
    echo "✓ Data generation completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Convert Data to SFT Format
# ===========================================

convert_data() {
    print_step "Converting Data to SFT Format (convert_data.sh)"
    
    cd $PROJECT_DIR
    
    # Check if converted data already exists
    if [ -f "$DATA_DIR/sft/train.json" ] && [ -f "$DATA_DIR/sft/val.json" ]; then
        echo "Converted SFT data already exists in $DATA_DIR/sft"
        echo -n "Reconvert data? (y/N): "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Skipping data conversion..."
            cd $SCRIPT_DIR
            return 0
        fi
    fi
    
    # Create data conversion script
    
    # Run conversion
    echo "Starting data conversion..."
    python convert_data_impl.py \
        --input_dir $DATA_DIR/raw \
        --output_dir $DATA_DIR/sft \
        --train_ratio 0.9 \
        --shuffle
    
    echo "✓ Data conversion completed"
    cd $SCRIPT_DIR
}

# ===========================================
# Create DeepSpeed Configuration
# ===========================================

create_deepspeed_configs() {
    print_step "Creating DeepSpeed Configurations"
    
    # Check if configs already exist
    if [ -f "$PROJECT_DIR/configs/ds_z3_config.json" ] && [ -f "$PROJECT_DIR/configs/ds_z2_config.json" ]; then
        echo "DeepSpeed configs already exist in $PROJECT_DIR/configs/"
        echo "Skipping DeepSpeed config creation..."
        return 0
    fi
    
    mkdir -p $PROJECT_DIR/configs
    
    # ZeRO-3 config for large models
    
    # ZeRO-2 config for faster training
    
# ===========================================
# Create Training Configuration
# ===========================================

create_training_config() {
    print_step "Creating LLaMA-Factory Training Configuration"
    
    mkdir -p $PROJECT_DIR/train_config
    
    # SFT training config

    # Full fine-tuning config
    
    # Create dataset info
    mkdir -p $PROJECT_DIR/data
    

    echo "✓ Training configs created at $PROJECT_DIR/train_config/"
}

# ===========================================
# Training Functions
# ===========================================

run_training() {
    print_step "Starting Model Training (train.sh)"
    
    cd $PROJECT_DIR
    
    # Set environment variables
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
    export WANDB_PROJECT="LLM4CardGame"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
    
    # Set GPU device (default to single GPU for LoRA training)
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    echo "Using GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # Get training config based on mode
    case $TRAINING_MODE in
        sft|lora)
            CONFIG_FILE="$PROJECT_DIR/train_config/sft_config.yaml"
            echo "Training mode: LoRA fine-tuning (single GPU)"
            ;;
        full)
            CONFIG_FILE="$PROJECT_DIR/train_config/full_config.yaml"
            echo "Training mode: Full fine-tuning"
            ;;
        *)
            echo "Unknown training mode: $TRAINING_MODE"
            echo "Available modes: sft, lora, full"
            exit 1
            ;;
    esac
    
    echo "Model: $BASE_MODEL"
    echo "GPUs: $NUM_GPUS"
    echo "Config: $CONFIG_FILE"
    echo ""
    
    # Create output directory
    mkdir -p $OUTPUT_DIR/logs
    
    # Run training with LLaMA-Factory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$OUTPUT_DIR/logs/train_${TIMESTAMP}.log"
    
    echo "Starting training at $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Training command - use single GPU for LoRA, multi-GPU only for full fine-tuning
    if [ $NUM_GPUS -gt 1 ] && [ "$TRAINING_MODE" = "full" ]; then
        # Multi-GPU training with DeepSpeed (only for full fine-tuning)
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        FORCE_TORCHRUN=1 NNODES=1 \
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    else
        # Single GPU training (default for LoRA)
        llamafactory-cli train $CONFIG_FILE 2>&1 | tee $LOG_FILE
    fi
    
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed successfully at $(date)"
        echo "Output directory: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check log file: $LOG_FILE"
    fi
    
    cd $SCRIPT_DIR
    return $TRAIN_EXIT_CODE
}

# ===========================================
# Continue Training with General Data
# ===========================================

run_continue_training() {
    print_step "Continue Training with General Data (train_ct.sh)"
    
    echo "This step fine-tunes the model with general data to prevent capability degradation."
    echo "Using mixed data: card games + general instruction following"
    
    # TODO: Implement continue training with general data
    # This would load the previous checkpoint and continue training
    # with a mixture of card game data and general instruction data
    
    echo "⚠️  Continue training not yet implemented"
    echo "Please refer to the original repository for implementation details"
}

# ===========================================
# Evaluation Functions
# ===========================================

run_evaluation() {
    print_step "Running Evaluation"
    
    cd $PROJECT_DIR
    
    # Create evaluation script
    cat > eval_impl.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate LLM performance on card games.
"""

import os
import json
import argparse
from pathlib import Path

def evaluate_on_game(model_path, game_name, num_games=100):
    """Evaluate model on a specific game."""
    print(f"Evaluating on {game_name}...")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Run evaluation
    wins = 0
    total = 0
    
    # TODO: Implement actual game evaluation
    # This would require setting up the game environment
    # and playing against the teacher model
    
    return {
        'game': game_name,
        'wins': wins,
        'total': total,
        'win_rate': wins / total if total > 0 else 0
    }

def evaluate_general(model_path, benchmark):
    """Evaluate on general benchmarks using OpenCompass."""
    print(f"Evaluating on {benchmark}...")
    
    # Run OpenCompass evaluation
    # opencompass --models $model_path --datasets $benchmark
    
    return {'benchmark': benchmark, 'score': 0}

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM4CardGame model')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--eval_type', type=str, default='all', 
                       choices=['game', 'general', 'all'], help='Evaluation type')
    parser.add_argument('--games', type=str, nargs='+', 
                       default=['doudizhu', 'mahjong', 'blackjack'],
                       help='Games to evaluate')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['mmlu_pro', 'math500', 'humaneval'],
                       help='General benchmarks')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    
    if args.eval_type in ['game', 'all']:
        results['game_eval'] = {}
        for game in args.games:
            result = evaluate_on_game(args.model_path, game)
            results['game_eval'][game] = result
    
    if args.eval_type in ['general', 'all']:
        results['general_eval'] = {}
        for benchmark in args.benchmarks:
            result = evaluate_general(args.model_path, benchmark)
            results['general_eval'][benchmark] = result
    
    # Save results
    output_file = Path(args.output_dir) / 'eval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
EOF

    echo "Evaluation types:"
    echo "  1. Game evaluation - Test against teacher models"
    echo "  2. General benchmarks - MMLU-Pro, Math-500, HumanEval"
    echo ""
    
    # Get model path
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    if [ ! -d "$MODEL_CHECKPOINT" ]; then
        echo "Warning: Model checkpoint not found at $MODEL_CHECKPOINT"
        echo "Please train a model first or specify MODEL_CHECKPOINT"
        return 1
    fi
    
    echo "Evaluating model: $MODEL_CHECKPOINT"
    
    python eval_impl.py \
        --model_path $MODEL_CHECKPOINT \
        --eval_type all \
        --output_dir $OUTPUT_DIR/eval_results
    
    cd $SCRIPT_DIR
}

# ===========================================
# Evaluation Scripts from Paper
# ===========================================

eval_checkpoint_scaling() {
    print_step "Evaluating Checkpoint Scaling (de_ckpt.sh)"
    echo "This evaluates how model performance changes with data volume"
    echo "Paper Question 1: How much data is required to master games?"
    
    # TODO: Implement checkpoint evaluation at different training steps
    echo "⚠️  Checkpoint scaling evaluation not yet implemented"
}

eval_mixture_model() {
    print_step "Evaluating Mixture Model (de_final.sh)"
    echo "This evaluates the mixture model on all 8 games"
    echo "Paper Question 2: Can LLMs simultaneously master multiple games?"
    
    run_evaluation
}

eval_api_models() {
    print_step "Evaluating API Models (eval_llm_one_on_all.sh)"
    echo "This evaluates API-based models (GPT-4, Claude, etc.) on card games"
    
    # TODO: Implement API model evaluation
    echo "⚠️  API model evaluation not yet implemented"
}

eval_general_benchmarks() {
    print_step "Evaluating General Benchmarks (eval_general.sh)"
    echo "This evaluates models on MMLU-Pro, Math-500, HumanEval"
    echo "Paper Question 3: Do models maintain general capabilities?"
    
    # Run OpenCompass evaluation
    echo "Running OpenCompass evaluation..."
    
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$OUTPUT_DIR/sft_lora}"
    
    # opencompass --models $MODEL_CHECKPOINT --datasets mmlu_pro math500 humaneval
    
    echo "⚠️  OpenCompass evaluation requires additional setup"
    echo "Please refer to: https://github.com/open-compass/opencompass"
}

# ===========================================
# Help and Usage
# ===========================================

show_help() {
    cat << 'EOF'
LLM4CardGame Training Pipeline

Usage: ./llm4cardgame_run.sh [command] [options]

Commands:
  setup           - Setup conda environment and dependencies
  clone           - Clone the repository
  generate        - Generate game data using teacher models
  convert         - Convert data to SFT format
  train           - Train the model
  eval            - Evaluate the model
  all             - Run complete pipeline (setup -> train -> eval)
  
  # Specific evaluation scripts from paper:
  eval_ckpt       - Evaluate checkpoint scaling (Question 1)
  eval_final      - Evaluate mixture model (Question 2)  
  eval_api        - Evaluate API models
  eval_general    - Evaluate general benchmarks (Question 3)

Options:
  BASE_MODEL      - Base model to fine-tune (default: Qwen/Qwen2.5-7B-Instruct)
  TRAINING_MODE   - Training mode: sft, lora, full (default: sft)
  NUM_GPUS        - Number of GPUs to use (default: auto-detect)

Examples:
  # Full pipeline with default settings
  ./llm4cardgame_run.sh all
  
  # Setup environment only
  ./llm4cardgame_run.sh setup
  
  # Train with specific model
  ./llm4cardgame_run.sh train meta-llama/Llama-3.1-8B-Instruct lora
  
  # Run evaluation
  MODEL_CHECKPOINT=./output/sft_lora ./llm4cardgame_run.sh eval

Environment Variables:
  PROJECT_DIR     - Project directory (default: ./LLM4CardGame)
  DATA_DIR        - Data directory (default: PROJECT_DIR/data)
  OUTPUT_DIR      - Output directory (default: PROJECT_DIR/output)
  NUM_EPISODES    - Episodes per game for data generation (default: 10000)
  WANDB_MODE      - WandB mode: online, offline, disabled (default: disabled)
  
Dependencies:
  - DouZero: https://github.com/kwai/DouZero (Doudizhu AI)
  - DanZero: https://github.com/submit-paper/Danzero_plus (Mahjong AI)
  - RLCard: https://github.com/datamllab/rlcard (Multi-game framework)
  - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory (Training)
  - OpenCompass: https://github.com/open-compass/opencompass (Evaluation)

Paper: "Can Large Language Models Master Complex Card Games?"
Repository: https://github.com/THUDM/LLM4CardGame
EOF
}

# ===========================================
# Main Execution
# ===========================================

main() {
    COMMAND=${1:-"help"}
    
    case $COMMAND in
        setup)
            check_system
            setup_environment
            ;;
        clone)
            clone_repository
            ;;
        generate|gen)
            generate_game_data
            ;;
        convert)
            convert_data
            ;;
        config)
            create_deepspeed_configs
            create_training_config
            ;;
        train)
            shift
            # Override BASE_MODEL and TRAINING_MODE if provided as arguments
            if [ -n "$1" ]; then
                BASE_MODEL="$1"
            fi
            if [ -n "$2" ]; then
                TRAINING_MODE="$2"
            fi
            create_training_config
            run_training
            ;;
        train_ct|continue)
            run_continue_training
            ;;
        eval|evaluate)
            run_evaluation
            ;;
        eval_ckpt)
            eval_checkpoint_scaling
            ;;
        eval_final)
            eval_mixture_model
            ;;
        eval_api)
            eval_api_models
            ;;
        eval_general)
            eval_general_benchmarks
            ;;
        all)
            check_system
            setup_environment
            clone_repository
            generate_game_data
            convert_data
            create_deepspeed_configs
            create_training_config
            run_training
            run_evaluation
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            echo "Use './llm4cardgame_run.sh help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"(base) jiacheng@ags1:/data/jiacheng/system/cache/temp/icml2026/walking$ 