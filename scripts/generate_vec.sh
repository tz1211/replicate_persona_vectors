gpu=${1:-0}


CUDA_VISIBLE_DEVICES=$gpu 
uv run python -m eval.eval_persona \
    --model Qwen/Qwen3-1.7B \
    --trait evil \
    --output_path output/eval_persona_extract/Qwen3-1.7B/evil_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name evil \
    --judge_model gpt-4.1-mini-2025-04-14 \
    --version extract

CUDA_VISIBLE_DEVICES=$gpu 
uv run python -m eval.eval_persona \
    --model Qwen/Qwen3-1.7B \
    --trait evil \
    --output_path output/eval_persona_extract/Qwen3-1.7B/evil_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --judge_model gpt-4.1-mini-2025-04-14 \
    --version extract

CUDA_VISIBLE_DEVICES=$gpu 
uv run python generate_vec.py \
    --model_name Qwen/Qwen3-1.7B \
    --pos_path output/eval_persona_extract/Qwen3-1.7B/evil_pos_instruct.csv \
    --neg_path output/eval_persona_extract/Qwen3-1.7B/evil_neg_instruct.csv \
    --trait evil \
    --save_dir persona_vectors/Qwen3-1.7B/ \
    --threshold 50