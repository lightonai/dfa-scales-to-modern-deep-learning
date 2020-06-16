import argparse
import torch
import yaml

from nerf import CfgNode, get_ray_bundle, load_blender_data, load_llff_data, models, meshgrid_xy, get_embedding_function, run_one_iter_of_nerf, img2mse, mse2psnr

def main():

    # Config options:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to (.yml) config file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint / pre-trained model to evaluate.")
    parser.add_argument("--dual-render", action='store_true', default=False)
    parser.add_argument('--gpu-id', type=int, default=0, help="id of the CUDA GPU to use (default: 0)")
    configargs = parser.parse_args()

    cfg = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    # Dataset:
    images, poses, render_poses, hwf = None, None, None, None
    i_train, i_val, i_test = None, None, None
    if cfg.dataset.type.lower() == "blender":
        images, poses, render_poses, hwf, i_split = load_blender_data(
            cfg.dataset.basedir,
            half_res=cfg.dataset.half_res,
            testskip=cfg.dataset.testskip,
            )
        i_train, i_val, i_test = i_split
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]
        if cfg.nerf.train.white_background:
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
    elif cfg.dataset.type.lower() == "llff":
        images, poses, bds, render_poses, i_test = load_llff_data(
            cfg.dataset.basedir, factor=cfg.dataset.downsample_factor,
        )
        hwf = poses[0, :3, -1]
        H, W, focal = hwf
        hwf = [int(H), int(W), focal]
        render_poses = torch.from_numpy(render_poses)
        images = torch.from_numpy(images)
        poses = torch.from_numpy(poses)

    # Hardware
    device = "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(configargs.gpu_id)
        device = "cuda"

    # Model
    encode_position_fn = get_embedding_function(num_encoding_functions=cfg.models.coarse.num_encoding_fn_xyz,
                                                include_input=cfg.models.coarse.include_input_xyz,
                                                log_sampling=cfg.models.coarse.log_sampling_xyz)

    encode_direction_fn = None
    if cfg.models.coarse.use_viewdirs:
        encode_direction_fn = get_embedding_function(num_encoding_functions=cfg.models.coarse.num_encoding_fn_dir,
                                                     include_input=cfg.models.coarse.include_input_dir,
                                                     log_sampling=cfg.models.coarse.log_sampling_dir)

    model_coarse = getattr(models, cfg.models.coarse.type)(num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
                                                           num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
                                                           include_input_xyz=cfg.models.coarse.include_input_xyz,
                                                           include_input_dir=cfg.models.coarse.include_input_dir,
                                                           use_viewdirs=cfg.models.coarse.use_viewdirs)
    model_coarse.to(device)

    model_fine = None
    if hasattr(cfg.models, "fine"):
        model_fine = getattr(models, cfg.models.fine.type)(num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
                                                           num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
                                                           include_input_xyz=cfg.models.fine.include_input_xyz,
                                                           include_input_dir=cfg.models.fine.include_input_dir,
                                                           use_viewdirs=cfg.models.fine.use_viewdirs)
        model_fine.to(device)

    # Load checkpoint
    checkpoint = torch.load(configargs.checkpoint, map_location=device)
    model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
    if checkpoint["model_fine_state_dict"]:
        try:
            model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        except:
            print("The checkpoint has a fine-level model, but it could "
                "not be loaded (possibly due to a mismatched config file.")
    if "height" in checkpoint.keys():
        hwf[0] = checkpoint["height"]
    if "width" in checkpoint.keys():
        hwf[1] = checkpoint["width"]
    if "focal_length" in checkpoint.keys():
        hwf[2] = checkpoint["focal_length"]


    # Prepare model and data
    model_coarse.eval()
    if model_fine:
        model_fine.eval()

    render_poses = render_poses.float().to(device)

    print("Dual render?", configargs.dual_render)

    # Evaluation loop
    with torch.no_grad():
        fine_psnrs = []
        if type(i_test) != list:
            i_test = [i_test]
        for i in i_test:
            print(f"Test sample {i + 1 - i_test[0]}/{i_test[-1] - i_test[0]}...")

            img_target = images[i].to(device)
            pose_target = poses[i, :3, :4].to(device)
            ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)
            rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(H, W, focal, model_coarse, model_fine, ray_origins,
                                                                    ray_directions, cfg, mode="validation",
                                                                    encode_position_fn=encode_position_fn,
                                                                    encode_direction_fn=encode_direction_fn,
                                                                    dual_render=configargs.dual_render)
            target_ray_values = img_target
            coarse_loss = img2mse(rgb_coarse[..., :3], target_ray_values[..., :3])
            loss, fine_loss = 0.0, 0.0
            if rgb_fine is not None:
                fine_loss = img2mse(rgb_fine[..., :3], target_ray_values[..., :3])
                loss = fine_loss
            else:
                loss = coarse_loss
            loss = coarse_loss + fine_loss
            psnr = mse2psnr(loss.item())
            psnr_coarse = mse2psnr(coarse_loss)
            psnr_fine = mse2psnr(fine_loss)
            print(f"\t Loss at sample: {psnr} (f:{psnr_fine}, c:{psnr_coarse})")

            fine_psnrs.append(psnr_fine)

        print(f"Validation PSNR: {sum(fine_psnrs) / len(fine_psnrs)}.")



if __name__ == "__main__":
    main()
