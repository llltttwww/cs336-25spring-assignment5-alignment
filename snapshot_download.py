from huggingface_hub import snapshot_download

# 模型 repo_id
repo_id = "Qwen/Qwen2.5-Math-1.5B"

# 下载到指定路径
snapshot_download(
    repo_id=repo_id,
    cache_dir="/workspace/home/luotianwei/hf_cache/hub",
    local_files_only=False,  # 如果你本地有缓存可以设 True
    resume_download=True
)

print(f"Model {repo_id} downloaded to /workspace/home/luotianwei/hf_cache/hub")