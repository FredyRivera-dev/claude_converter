from claude_converter import Converter
from huggingface_hub import hf_hub_download

converter = Converter()

hf_hub_download(repo_id="armand0e/claude-fable-5-claude-code", 
        filename="06ec42c3-2184-40c5-b0ee-98c3235b4c4c.jsonl", 
        repo_type="dataset",
        local_dir=".")

converter.inspect_session("06ec42c3-2184-40c5-b0ee-98c3235b4c4c.jsonl")