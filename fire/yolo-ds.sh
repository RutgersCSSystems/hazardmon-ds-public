source scripts/setvars.sh
mkdir confuse_data 
cd confuse_data

pip install gdown
export PATH="$HOME/.local/bin:$PATH"

dataset_1='1TQKA9nzo0BVwtmojmSusDt5j02KWzIu9' # Google drive item ID
# Google drive item ID
# ROBOFLOW: https://public.roboflow.com/project/wildfire-smoke/1
dataset_2='1cC5gNkDsOlgRl7f4Vm8VeaHj3LDdvTlt' 

gdown $dataset_1
#gdown $dataset_2 # Choose dataset 1 or 2
mv fire.zip confuse_data.zip
unzip confuse_data.zip #//You will see train and val folder
