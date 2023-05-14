export PYTHONPATH=../../:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1
python -u ../../run.py -c custom_vae_mask.yaml &
