
cd /



cd ~/Projets/git/




mount_path="$1"
json_path="$2"
generated_data_path="$3"
validation_folder="$4"
singularity exec -e -B "$mount_path":/Transformer_pommiers VPlants2.simg bash /Transformer_pommiers/singularity_workspace/script.sh "$json_path" "$generated_data_path" "$validation_folder"