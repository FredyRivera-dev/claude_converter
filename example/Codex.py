from claude_converter import Converter
from huggingface_hub import hf_hub_download

converter = Converter(converter="codex")

hf_hub_download(repo_id="AletheiaResearch/GPT-5.5-Codex", 
        filename="rollout-2026-06-22T08-33-58-019eee77-052d-7530-af09-17a140e08123.jsonl", 
        repo_type="dataset",
        local_dir=".")

converter.inspect_session("rollout-2026-06-22T08-33-58-019eee77-052d-7530-af09-17a140e08123.jsonl")