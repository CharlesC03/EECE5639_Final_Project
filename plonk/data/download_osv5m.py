from huggingface_hub import snapshot_download
snapshot_download(repo_id="osv5m/osv5m-wds", local_dir="plonk/datasets/osv5m", repo_type='dataset')
