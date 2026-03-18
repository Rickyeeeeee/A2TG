import argparse
import itertools
import math
import os
import shlex
import sys
from pathlib import Path


# POINT_COUNTS = [10000, 50000, 100000, 500000, 1000000]
POINT_COUNTS = [500000, 1000000]
MAX_STEPS = 30000
CKPT_STEP = 29999

DB_SCENES = ["drjohnson", "playroom"]
TANDT_SCENES = ["train", "truck"]
MIPNERF360_OUTDOOR_SCENES = ["bicycle", "garden", "stump"]
MIPNERF360_INDOOR_SCENES = ["room", "counter", "kitchen", "bonsai"]
MIPNERF360_SCENES = MIPNERF360_OUTDOOR_SCENES + MIPNERF360_INDOOR_SCENES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal fixed-point-count full eval launcher for db/tandt/mipnerf360."
    )
    parser.add_argument("--db-dataset", required=True, help="db dataset root path.")
    parser.add_argument("--tandt-dataset", required=True, help="tandt dataset root path.")
    parser.add_argument("--mipnerf360-dataset", required=True, help="MipNeRF360 dataset root path.")
    parser.add_argument("--output", required=True, help="Output root path.")
    parser.add_argument("--cuda-device-id", type=int, default=0)
    parser.add_argument("--port", type=int, default=6070)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    output_root = Path(args.output).expanduser()

    datasets = [
        ("db", Path(args.db_dataset).expanduser(), DB_SCENES),
        ("tandt", Path(args.tandt_dataset).expanduser(), TANDT_SCENES),
        ("mipnerf360", Path(args.mipnerf360_dataset).expanduser(), MIPNERF360_SCENES),
    ]

    for dataset, dataset_root, scenes in datasets:
        print(f"[INFO] Running dataset {dataset} from {dataset_root}")

        dataset_output_root = output_root / dataset

        for scene, point_count in itertools.product(scenes, POINT_COUNTS):
            scene_dir = dataset_root / scene

            if dataset in ("db", "tandt"):
                data_factor = 1
            elif scene in MIPNERF360_OUTDOOR_SCENES:
                data_factor = 4
            else:
                data_factor = 2

            scene_dir_q = shlex.quote(str(scene_dir))
            common_args = (
                f"--dataset colmap --init_num_pts {point_count} --strategy.cap-max {point_count} "
                f"--port {args.port} --disable_viewer --data_factor {data_factor}"
            )

            dgs_result_dir = dataset_output_root / "2dgs_mcmc" / f"pc{point_count}" / scene
            dgs_result_dir_q = shlex.quote(str(dgs_result_dir))
            cmd_2dgs = (
                f"CUDA_VISIBLE_DEVICES={args.cuda_device_id} python trainer.py mcmc "
                f"--max_steps {MAX_STEPS} --eval_steps 30000 --save_steps 30000 "
                f"--data_dir {scene_dir_q} --result_dir {dgs_result_dir_q} "
                f"--init_extent 1 --init_type sfm --model_type=2dgs "
                f"{common_args} --upscale_start_iter 100000000"
            )
            print(f"[INFO] Running command for {dataset}/{scene}, pc={point_count}: {cmd_2dgs}")
            code = os.system(cmd_2dgs)
            if code != 0:
                return code

            base_ckpt = (
                dataset_output_root / "2dgs_mcmc" / f"pc{point_count}" / scene / "ckpts" / f"ckpt_{CKPT_STEP}.pt"
            )
            base_ckpt_q = shlex.quote(str(base_ckpt))

            a2tg_result_dir = dataset_output_root / "a2tg" / f"pc{point_count}" / scene
            a2tg_result_dir_q = shlex.quote(str(a2tg_result_dir))
            cmd_a2tg = (
                f"CUDA_VISIBLE_DEVICES={args.cuda_device_id} python trainer.py mcmc "
                f"--data_dir {scene_dir_q} --pretrained_path {base_ckpt_q} --result_dir {a2tg_result_dir_q} "
                f"--eval_steps 30000 --save_steps 30000 --init_type pretrained --model_type=a2tg "
                f"{common_args} --strategy.refine-start-iter=1000000000000 "
                f"--textured_rgb --textured_alpha --texture_resolution 1 "
                f"--min_aspect_ratio=4.0 --max_scale_for_thin=0.01 --upscale_grad2d=0.00002 "
                f"--upscale_start_iter=0 --upscale_stop_iter={500 * int(math.log2(4)) + 2} --upscale_every=500"
            )
            print(f"[INFO] Running command for {dataset}/{scene}, pc={point_count}: {cmd_a2tg}")
            code = os.system(cmd_a2tg)
            if code != 0:
                return code

            if False:
                tg_result_dir = dataset_output_root / "textured_gaussians_rgba" / f"pc{point_count}" / scene
                tg_result_dir_q = shlex.quote(str(tg_result_dir))
                cmd_tg = (
                    f"CUDA_VISIBLE_DEVICES={args.cuda_device_id} python trainer.py mcmc "
                    f"--data_dir {scene_dir_q} --pretrained_path {base_ckpt_q} --result_dir {tg_result_dir_q} "
                    f"--eval_steps 30000 --save_steps 30000 --init_type pretrained --model_type=textured_gaussians "
                    f"{common_args} --strategy.refine-start-iter=1000000000000 "
                    f"--textured_rgb --textured_alpha --texture_resolution 4 "
                    f"--upscale_grad2d=0.00002 --upscale_start_iter=1000000 "
                    f"--upscale_stop_iter=1002 --upscale_every=500"
                )
                print(f"[INFO] Running command for {dataset}/{scene}, pc={point_count}: {cmd_tg}")
                code = os.system(cmd_tg)
                if code != 0:
                    return code

    print("[INFO] All jobs finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
