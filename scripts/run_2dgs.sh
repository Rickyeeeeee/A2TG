cd ../examples
export CUDA_VISIBLE_DEVICES=$1
results_dir="../results/2dgs"
python trainer.py mcmc \
    --data_dir "/miele/brian/data/nerf_synthetic/lego/" \
    --result_dir "../results/2dgs/lego" \
    --dataset "blender" \
    --init_extent 1 \
    --init_type "random" \
    --background_mode "white" \
    --model_type=2dgs \
    --init_num_pts=10000 \
    --strategy.cap-max=10000 \
    --alpha_loss \
    --port 6070