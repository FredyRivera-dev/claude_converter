from claude_converter import Converter
from huggingface_hub import hf_hub_download

converter = Converter(converter="pi")

hf_hub_download(repo_id="armand0e/claude-opus-4.8-pi-traces", 
        filename="2026-06-07T00-07-46-038Z_019e9f68-3075-7136-b429-c6b2c871ed67.jsonl", 
        repo_type="dataset",
        local_dir=".")

converter.inspect_session("2026-06-07T00-07-46-038Z_019e9f68-3075-7136-b429-c6b2c871ed67.jsonl")