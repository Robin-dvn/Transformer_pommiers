cd ~/Projets/git/
json_path="$1"
generated_data_path="$2"
validation_folder="$3"
singularity exec -e -B /mnt/c/Users/Robin/Documents/Stage\ pommiers/Transformer_pommiers:/Transformer_pommiers VPlants2.simg bash /Transformer_pommiers/singularity_workspace/script.sh "$json_path" "$generated_data_path" "$validation_folder"