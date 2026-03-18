cd ../examples
export CUDA_VISIBLE_DEVICES=$1
results_dir="../results/textured_gaussians_rgba"
python trainer.py mcmc \
    --data_dir "/miele/brian/data/nerf_synthetic/chair/" \
    --pretrained_path "../results/2dgs/chair/ckpts/ckpt_29999.pt" \
    --result_dir "${results_dir}/chair" \
    --dataset "blender" \
    --init_extent 1 \
    --init_type "pretrained" \
    --background_mode "white" \
    --model_type=textured_gaussians \
    --init_num_pts=10000 \
    --strategy.cap-max=10000 \
    --strategy.refine-start-iter=1000000000000 \
    --alpha_loss \
    --textured_rgb \
    --textured_alpha \
    --port 6070