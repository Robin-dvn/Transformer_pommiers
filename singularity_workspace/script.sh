cd /Transformer_pommiers/singularity_workspace
python -m pip install tqdm==4.11.2
json_path="$1"
generated_path="$2"
validation_folder="$3"
python main.py "$json_path" "$generated_path" "$validation_folder"