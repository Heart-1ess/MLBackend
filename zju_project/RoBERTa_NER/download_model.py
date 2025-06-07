from huggingface_hub import snapshot_download

snapshot_download(repo_id="hfl/chinese-roberta-wwm-ext", repo_type="model", cache_dir="/home/zhangshuhao/projects/ys/label-studio-ml-backend/zju_project/RoBERTa_NER", local_dir="/home/zhangshuhao/projects/ys/label-studio-ml-backend/zju_project/RoBERTa_NER")