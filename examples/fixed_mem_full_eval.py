import argparse
import itertools
import math
import os
import shlex
import sys
from pathlib import Path


MAX_STEPS = 30000
CKPT_STEP = 29999

DB_SCENES = ["drjohnson", "playroom"]
TANDT_SCENES = ["train", "truck"]
MIPNERF360_OUTDOOR_SCENES = ["bicycle", "garden", "stump"]
MIPNERF360_INDOOR_SCENES = ["room", "counter", "kitchen", "bonsai"]
MIPNERF360_SCENES = MIPNERF360_OUTDOOR_SCENES + MIPNERF360_INDOOR_SCENES

SPLAT_SIZES = {
    "2dgs": 58,
    "textured_gaussians": 122,
}

A2TG_POINT_COUNTS = {
    "db": {60: 210000, 100: 340000, 200: 700000},
    "tandt": {50: 160000, 100: 340000, 200: 690000},
    "mipnerf360": {60: 200000, 100: 340000, 200: 700000},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-memory full eval launcher for db/tandt/mipnerf360."
    )
    parser.add_argument("--mem", type=int, default=200, help="Target memory budget in MB.")
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
    trainer_script = Path(__file__).resolve().with_name("trainer.py")
    if not trainer_script.is_file():
        raise FileNotFoundError(f"trainer.py not found: {trainer_script}")

    py = shlex.quote(sys.executable)
    trainer = shlex.quote(str(trainer_script))
    a2tg_upscale_stop_iter = 500 * int(math.log2(4)) + 2

    point_count_2dgs = int(float(args.mem) * 1000.0 * 1000.0 / (SPLAT_SIZES["2dgs"] * 4))
    point_count_tg = int(
        float(args.mem) * 1000.0 * 1000.0 / (SPLAT_SIZES["textured_gaussians"] * 4)
    )

    dataset_runs = [
        ("db", Path(args.db_dataset).expanduser(), DB_SCENES),
        ("tandt", Path(args.tandt_dataset).expanduser(), TANDT_SCENES),
        ("mipnerf360", Path(args.mipnerf360_dataset).expanduser(), MIPNERF360_SCENES),
    ]

    for preset, dataset_root, scenes in dataset_runs:
        if not dataset_root.exists():
            raise FileNotFoundError(f"{preset} dataset path does not exist: {dataset_root}")
        if not dataset_root.is_dir():
            raise NotADirectoryError(f"{preset} dataset path is not a directory: {dataset_root}")
        if args.mem not in A2TG_POINT_COUNTS[preset]:
            raise ValueError(
                f"Unsupported --mem {args.mem} for {preset} a2tg run. "
                f"Supported: {sorted(A2TG_POINT_COUNTS[preset].keys())}"
            )

        point_count_a2tg = A2TG_POINT_COUNTS[preset][args.mem]
        dataset_output_root = output_root / preset

        print(f"[INFO] Running preset {preset} from {dataset_root}")
        print(f"[INFO] 2dgs memory point count: {point_count_2dgs}")
        print(f"[INFO] textured_gaussians memory point count: {point_count_tg}")
        print(f"[INFO] a2tg memory point count: {point_count_a2tg}")

        # Phase 1: 2DGS at fixed memory for 2DGS
        for scene in scenes:
            scene_dir = dataset_root / scene
            if not scene_dir.is_dir():
                raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

            if preset in ("db", "tandt"):
                data_factor = 1
            elif scene in MIPNERF360_OUTDOOR_SCENES:
                data_factor = 4
            else:
                data_factor = 2

            scene_dir_q = shlex.quote(str(scene_dir))
            common_args = (
                f"--dataset colmap --init_num_pts {point_count_2dgs} --strategy.cap-max {point_count_2dgs} "
                f"--port {args.port} --disable_viewer --data_factor {data_factor}"
            )

            dgs_result_dir = dataset_output_root / "2dgs_mcmc" / f"pc{point_count_2dgs}" / scene
            dgs_result_dir_q = shlex.quote(str(dgs_result_dir))
            cmd_2dgs = (
                f"CUDA_VISIBLE_DEVICES={args.cuda_device_id} {py} {trainer} mcmc "
                f"--max_steps {MAX_STEPS} --eval_steps 30000 --save_steps 30000 "
                f"--data_dir {scene_dir_q} --result_dir {dgs_result_dir_q} "
                f"--init_extent 1 --init_type sfm --model_type=2dgs "
                f"{common_args} --upscale_start_iter 100000000"
            )
            print(f"[INFO] Running command for {preset}/{scene} [2dgs-mem], pc={point_count_2dgs}: {cmd_2dgs}")
            code = os.system(cmd_2dgs)
            if code != 0:
                return code

        # Phase 2: 2DGS + textured_gaussians at fixed memory for textured_gaussians
        for scene in scenes:
            scene_dir = dataset_root / scene
            if preset in ("db", "tandt"):
                data_factor = 1
            elif scene in MIPNERF360_OUTDOOR_SCENES:
                data_factor = 4
            else:
                data_factor = 2

            scene_dir_q = shlex.quote(str(scene_dir))
            common_args = (
                f"--dataset colmap --init_num_pts {point_count_tg} --strategy.cap-max {point_count_tg} "
                f"--port {args.port} --disable_viewer --data_factor {data_factor}"
            )

            dgs_result_dir = dataset_output_root / "2dgs_mcmc" / f"pc{point_count_tg}" / scene
            dgs_result_dir_q = shlex.quote(str(dgs_result_dir))
            cmd_2dgs = (
                f"CUDA_VISIBLE_DEVICES={args.cuda_device_id} {py} {trainer} mcmc "
                f"--max_steps {MAX_STEPS} --eval_steps 30000 --save_steps 30000 "
                f"--data_dir {scene_dir_q} --result_dir {dgs_result_dir_q} "
                f"--init_extent 1 --init_type sfm --model_type=2dgs "
                f"{common_args} --upscale_start_iter 100000000"
            )
            print(f"[INFO] Running command for {preset}/{scene} [tg-mem 2dgs], pc={point_count_tg}: {cmd_2dgs}")
            code = os.system(cmd_2dgs)
            if code != 0:
                return code

            base_ckpt = (
                dataset_output_root
                / "2dgs_mcmc"
                / f"pc{point_count_tg}"
                / scene
                / "ckpts"
                / f"ckpt_{CKPT_STEP}.pt"
            )
            base_ckpt_q = shlex.quote(str(base_ckpt))

            tg_result_dir = dataset_output_root / "textured_gaussians_rgba" / f"pc{point_count_tg}" / scene
            tg_result_dir_q = shlex.quote(str(tg_result_dir))
            cmd_tg = (
                f"CUDA_VISIBLE_DEVICES={args.cuda_device_id} {py} {trainer} mcmc "
                f"--data_dir {scene_dir_q} --pretrained_path {base_ckpt_q} --result_dir {tg_result_dir_q} "
                f"--dataset colmap --init_type pretrained --model_type=textured_gaussians "
                f"{common_args} --strategy.refine-start-iter=1000000000000 "
                f"--textured_rgb --textured_alpha --texture_resolution 4 "
                f"--upscale_grad2d=0.00002 --upscale_start_iter=1000000 "
                f"--upscale_stop_iter=1002 --upscale_every=500"
            )
            print(f"[INFO] Running command for {preset}/{scene} [tg-mem textured], pc={point_count_tg}: {cmd_tg}")
            code = os.system(cmd_tg)
            if code != 0:
                return code

        # Phase 3: 2DGS + a2tg at fixed memory for a2tg
        for scene in scenes:
            scene_dir = dataset_root / scene
            if preset in ("db", "tandt"):
                data_factor = 1
            elif scene in MIPNERF360_OUTDOOR_SCENES:
                data_factor = 4
            else:
                data_factor = 2

            scene_dir_q = shlex.quote(str(scene_dir))
            common_args = (
                f"--dataset colmap --init_num_pts {point_count_a2tg} --strategy.cap-max {point_count_a2tg} "
                f"--port {args.port} --disable_viewer --data_factor {data_factor}"
            )

            dgs_result_dir = dataset_output_root / "2dgs_mcmc" / f"pc{point_count_a2tg}" / scene
            dgs_result_dir_q = shlex.quote(str(dgs_result_dir))
            cmd_2dgs = (
                f"CUDA_VISIBLE_DEVICES={args.cuda_device_id} {py} {trainer} mcmc "
                f"--max_steps {MAX_STEPS} --eval_steps 30000 --save_steps 30000 "
                f"--data_dir {scene_dir_q} --result_dir {dgs_result_dir_q} "
                f"--init_extent 1 --init_type sfm --model_type=2dgs "
                f"{common_args} --upscale_start_iter 100000000"
            )
            print(f"[INFO] Running command for {preset}/{scene} [a2tg-mem 2dgs], pc={point_count_a2tg}: {cmd_2dgs}")
            code = os.system(cmd_2dgs)
            if code != 0:
                return code

            base_ckpt = (
                dataset_output_root
                / "2dgs_mcmc"
                / f"pc{point_count_a2tg}"
                / scene
                / "ckpts"
                / f"ckpt_{CKPT_STEP}.pt"
            )
            base_ckpt_q = shlex.quote(str(base_ckpt))

            a2tg_result_dir = dataset_output_root / "a2tg" / f"pc{point_count_a2tg}" / scene
            a2tg_result_dir_q = shlex.quote(str(a2tg_result_dir))
            cmd_a2tg = (
                f"CUDA_VISIBLE_DEVICES={args.cuda_device_id} {py} {trainer} mcmc "
                f"--data_dir {scene_dir_q} --pretrained_path {base_ckpt_q} --result_dir {a2tg_result_dir_q} "
                f"--dataset colmap --init_type pretrained --model_type=a2tg "
                f"{common_args} --strategy.refine-start-iter=1000000000000 "
                f"--textured_rgb --textured_alpha --texture_resolution 1 "
                f"--min_aspect_ratio=4.0 --max_scale_for_thin=0.01 --upscale_grad2d=0.00002 "
                f"--upscale_start_iter=0 --upscale_stop_iter={a2tg_upscale_stop_iter} --upscale_every=500"
            )
            print(f"[INFO] Running command for {preset}/{scene} [a2tg-mem a2tg], pc={point_count_a2tg}: {cmd_a2tg}")
            code = os.system(cmd_a2tg)
            if code != 0:
                return code

    print("[INFO] All jobs finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
