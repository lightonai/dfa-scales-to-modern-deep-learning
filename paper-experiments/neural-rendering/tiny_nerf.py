import argparse
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm, trange

from tinydfa import DFA, DFALayer, FeedbackPointsHandling
import tinydfa.light_dfa as light_dfa

from nerf import cumprod_exclusive, get_minibatches, get_ray_bundle, positional_encoding


def compute_query_points_from_rays(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near_thresh: float,
    far_thresh: float,
    num_samples: int,
    randomize: Optional[bool] = True,
) -> (torch.Tensor, torch.Tensor):
    r"""Compute query 3D points given the "bundle" of rays. The near_thresh and far_thresh
    variables indicate the bounds within which 3D points are to be sampled.

    Args:
        ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
          `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
        ray_directions (torch.Tensor): Direction of each ray in the "bundle" as returned by the
          `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
        near_thresh (float): The 'near' extent of the bounding volume (i.e., the nearest depth
          coordinate that is of interest/relevance).
        far_thresh (float): The 'far' extent of the bounding volume (i.e., the farthest depth
          coordinate that is of interest/relevance).
        num_samples (int): Number of samples to be drawn along each ray. Samples are drawn
          randomly, whilst trying to ensure "some form of" uniform spacing among them.
        randomize (optional, bool): Whether or not to randomize the sampling of query points.
          By default, this is set to `True`. If disabled (by setting to `False`), we sample
          uniformly spaced points along each ray in the "bundle".

    Returns:
        query_points (torch.Tensor): Query points along each ray
          (shape: :math:`(width, height, num_samples, 3)`).
        depth_values (torch.Tensor): Sampled depth values along each ray
          (shape: :math:`(num_samples)`).
    """
    # TESTED
    # shape: (num_samples)
    depth_values = torch.linspace(near_thresh, far_thresh, num_samples).to(ray_origins)
    if randomize is True:
        # ray_origins: (width, height, 3)
        # noise_shape = (width, height, num_samples)
        noise_shape = list(ray_origins.shape[:-1]) + [num_samples]
        # depth_values: (num_samples)
        depth_values = (
            depth_values
            + torch.rand(noise_shape).to(ray_origins)
            * (far_thresh - near_thresh)
            / num_samples
        )
    # (width, height, num_samples, 3) = (width, height, 1, 3) + (width, height, 1, 3) * (num_samples, 1)
    # query_points:  (width, height, num_samples, 3)
    query_points = (
        ray_origins[..., None, :]
        + ray_directions[..., None, :] * depth_values[..., :, None]
    )
    # TODO: Double-check that `depth_values` returned is of shape `(num_samples)`.
    return query_points, depth_values


def render_volume_density(
    radiance_field: torch.Tensor, ray_origins: torch.Tensor, depth_values: torch.Tensor
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    r"""Differentiably renders a radiance field, given the origin of each ray in the
    "bundle", and the sampled depth values along them.

    Args:
    radiance_field (torch.Tensor): A "field" where, at each query location (X, Y, Z),
      we have an emitted (RGB) color and a volume density (denoted :math:`\sigma` in
      the paper) (shape: :math:`(width, height, num_samples, 4)`).
    ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
      `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
    depth_values (torch.Tensor): Sampled depth values along each ray
      (shape: :math:`(num_samples)`).

    Returns:
    rgb_map (torch.Tensor): Rendered RGB image (shape: :math:`(width, height, 3)`).
    depth_map (torch.Tensor): Rendered depth image (shape: :math:`(width, height)`).
    acc_map (torch.Tensor): # TODO: Double-check (I think this is the accumulated
      transmittance map).
    """
    # TESTED
    sigma_a = torch.nn.functional.relu(radiance_field[..., 3])
    rgb = torch.sigmoid(radiance_field[..., :3])
    one_e_10 = torch.tensor([1e10], dtype=ray_origins.dtype, device=ray_origins.device)
    dists = torch.cat(
        (
            depth_values[..., 1:] - depth_values[..., :-1],
            one_e_10.expand(depth_values[..., :1].shape),
        ),
        dim=-1,
    )
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

    rgb_map = (weights[..., None] * rgb).sum(dim=-2)
    depth_map = (weights * depth_values).sum(dim=-1)
    acc_map = weights.sum(-1)

    return rgb_map, depth_map, acc_map


# One iteration of TinyNeRF (forward pass).
def run_one_iter_of_tinynerf(
    height,
    width,
    focal_length,
    tform_cam2world,
    near_thresh,
    far_thresh,
    depth_samples_per_ray,
    encoding_function,
    get_minibatches_function,
    chunksize,
    model,
    encoding_function_args,
):

    # Get the "bundle" of rays through all image pixels.
    ray_origins, ray_directions = get_ray_bundle(
        height, width, focal_length, tform_cam2world
    )

    # Sample query points along each ray
    query_points, depth_values = compute_query_points_from_rays(
        ray_origins, ray_directions, near_thresh, far_thresh, depth_samples_per_ray
    )

    # "Flatten" the query points.
    flattened_query_points = query_points.reshape((-1, 3))

    # Encode the query points (default: positional encoding).
    encoded_points = encoding_function(flattened_query_points, encoding_function_args)

    # Split the encoded points into "chunks", run the model on all chunks, and
    # concatenate the results (to avoid out-of-memory issues).
    batches = get_minibatches_function(encoded_points, chunksize=chunksize)
    predictions = []
    for batch in batches:
        predictions.append(model(batch))
    radiance_field_flattened = torch.cat(predictions, dim=0)

    # "Unflatten" to obtain the radiance field.
    unflattened_shape = list(query_points.shape[:-1]) + [4]
    radiance_field = torch.reshape(radiance_field_flattened, unflattened_shape)

    # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_predicted, _, _ = render_volume_density(
        radiance_field, ray_origins, depth_values
    )

    return rgb_predicted


class VeryTinyNerfModel(torch.nn.Module):
    r"""Define a "very tiny" NeRF model comprising three fully connected layers.
    """

    def __init__(self, filter_size=128, num_encoding_functions=6, training_method='BP', feedback_mode=None):
        super(VeryTinyNerfModel, self).__init__()
        self.feedback_mode = feedback_mode
        if self.feedback_mode == 'HOOK':
            dfa_layer = light_dfa.DFALayer
            dfa = light_dfa.DFA
        else:
            dfa_layer = DFALayer
            dfa = DFA
        # Input layer (default: 39 -> 128)
        self.layer1 = torch.nn.Linear(3 + 3 * 2 * num_encoding_functions, filter_size)
        # Layer 2 (default: 128 -> 128)
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        # Layer 3 (default: 128 -> 4)
        self.layer3 = torch.nn.Linear(filter_size, 4)
        # Short hand for torch.nn.functional.relu
        self.relu = torch.nn.functional.relu

        self.training_method = training_method

        if self.training_method in ['DFA', 'SHALLOW']:
            self.dfa_1 = dfa_layer(name='dfa1')
            self.dfa_2 = dfa_layer(name='dfa2')
            if self.feedback_mode == 'HOOK':
                self.dfa = dfa([self.dfa_1, self.dfa_2], no_training= self.training_method=='SHALLOW')
            else:
                self.dfa = dfa([self.dfa_1, self.dfa_2], no_training= self.training_method == 'SHALLOW',
                               feedback_points_handling=FeedbackPointsHandling.MINIBATCH)

    def forward(self, x):
        if self.training_method in ['DFA', 'SHALLOW']:
            x = self.dfa_1(self.relu(self.layer1(x)))
            x = self.dfa_2(self.relu(self.layer2(x)))
            x = self.dfa(self.layer3(x))
        else:
            x = self.relu(self.layer1(x))
            x = self.relu(self.layer2(x))
            x = self.layer3(x)
        return x


def main(args):
    # Setup general stuff:
    device = torch.device(f"cuda:{args.gpu_id}")  # Setup CUDA GPU

    seed = 9458
    torch.manual_seed(seed)
    np.random.seed(seed)

    save_path = os.path.join(args.save_path, f'tiny-{args.training_method}-{args.feedback_mode}/')
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Prepare data from scene:
    data = np.load(os.path.join(args.dataset_path, "tiny_nerf_data.npz"))

    tform_cam2world = data["poses"]  # Camera poses
    tform_cam2world = torch.from_numpy(tform_cam2world).to(device)

    focal_length = data["focal"]  # Focal lengths
    focal_length = torch.from_numpy(focal_length).to(device)

    images = data["images"]  # Actual images
    height, width = images.shape[1:3]

    testimg, testpose = images[101], tform_cam2world[101]  # Test view
    testimg = torch.from_numpy(testimg).to(device)

    images = torch.from_numpy(images[:100, ..., :3]).to(device)


    # Training parameters:
    near_thresh = 2.0  # Near clipping treshold for depth
    far_thresh = 6.0  # Far clipping treshold for depth
    encode = positional_encoding  # Position and angle encoding function
    num_encoding_functions = 10  # Encoding depth
    depth_samples_per_ray = 32  # Samples along ray

    chunksize = 8192  # Numbers of rays queried at once (4096 ~ 1.4GB of GPU/ray/8 samples

    lr = 5e-3  # Learning rate
    num_iters = 5000  # Number of iterations

    display_every = 100  # Display stats every

    # Model:
    model = VeryTinyNerfModel(num_encoding_functions=num_encoding_functions, training_method=args.training_method,
                              feedback_mode=args.feedback_mode)
    model.to(device)

    # Optimizer:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop:
    psnrs = []  # Log PSNR values
    iternums = []  # Log number of iterations

    for i in trange(num_iters):
        # Randomly pick an image as the target.
        target_img_idx = np.random.randint(images.shape[0])
        target_img = images[target_img_idx].to(device)
        target_tform_cam2world = tform_cam2world[target_img_idx].to(device)

        # Run one iteration of TinyNeRF and get the rendered RGB image.
        rgb_predicted = run_one_iter_of_tinynerf(height, width, focal_length, target_tform_cam2world, near_thresh,
                                                 far_thresh, depth_samples_per_ray, encode, get_minibatches,
                                                 chunksize, model, num_encoding_functions,)

        # Compute mean-squared error between the predicted and target images, and backprop.
        loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Validate every so often:
        if i % display_every == 0 or i == num_iters - 1:
            # Render the held-out view and compute metrics on it:
            rgb_predicted = run_one_iter_of_tinynerf(height, width, focal_length, testpose, near_thresh, far_thresh,
                                                     depth_samples_per_ray, encode, get_minibatches, chunksize,
                                                     model, num_encoding_functions,)
            loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
            tqdm.write("Loss: " + str(loss.item()))
            psnr = -10.0 * torch.log10(loss)

            psnrs.append(psnr.item())  # Log the metrics
            iternums.append(i)

            plt.imshow(rgb_predicted.detach().cpu().numpy())
            plt.savefig(os.path.join(save_path, str(i).zfill(6) + ".png"))
            plt.close("all")

            if i == num_iters - 1:
                plt.plot(iternums, psnrs)
                plt.savefig(os.path.join(save_path, "psnr.png"))
                plt.close("all")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tiny-NeRF DFA example')
    parser.add_argument('-t', '--training-method', type=str, choices=['BP', 'DFA', 'SHALLOW'], default='SHALLOW',
                        metavar='T', help='training method to use, choose from backpropagation (BP), direct feedback '
                                          'alignment (DFA), or only topmost layer (SHALLOW) (default: BP)')
    parser.add_argument('-f', '--feedback-mode', type=str, choices=['MINIBATCH', 'HOOK'],
                        default='MINIBATCH', metavar='F', help='feedback handling mode to use when training with DFA.')

    parser.add_argument('-g', '--gpu-id', type=int, default='0', metavar='i',
                        help='id of the CUDA GPU to use (default: 0)')

    parser.add_argument('-p', '--dataset-path', type=str, default='/data', metavar='DP',
                        help='path to dataset (default: /data)')
    parser.add_argument('-s', '--save-path', type=str, default='/logs', metavar='SP',
                        help='path to saving folder (logs and outputs) (default: /logs)')
    args = parser.parse_args()
    main(args)
