gpu=${1:-0}


CUDA_VISIBLE_DEVICES=$gpu 
uv run python -m eval.cal_projection \
    --file_path output/eval_persona_eval/Qwen3-1.7B/evil.csv \
    --vector_path output/persona_vectors/Qwen3-1.7B/evil_response_avg_diff.pt \
    --layer 20 \
    --model_name Qwen/Qwen3-1.7B \
    --projection_type proj