"""
Modified renderer: Supports separate rendering of static and dynamic Gaussians
Added render_mode parameter to control rendering mode
"""
import math

import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
    compute_shortest_axis_view,
)

from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel
from thirdparty.gaussian_splatting.utils.sh_utils import eval_sh


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    mask=None,
    dynamic=False,
    dx=None,
    ds=None,
    dr=None,
    do=None,
    dc=None,
    render_mode='all',  # New parameter: 'all', 'static_only', 'dynamic_only'
    return_normal=False,  # When True, also render normal map (shortest axis weighted by opacity)
):
    """
    Render the scene with support for separate static/dynamic rendering.
    
    Args:
        viewpoint_camera: Camera viewpoint
        pc: GaussianModel - Gaussian model
        pipe: Rendering pipeline parameters
        bg_color: Background color (must be on GPU!)
        scaling_modifier: Scaling modifier
        override_color: Override color
        mask: Custom mask (optional)
        dynamic: Whether to use dynamic deformation
        dx, ds, dr, do, dc: Dynamic deformation parameters
        render_mode: Rendering mode
            - 'all': Render all Gaussians (default, backward compatible)
            - 'static_only': Render only static Gaussians
            - 'dynamic_only': Render only dynamic Gaussians
        return_normal: If True, add 'normal' to output (shortest-axis normal map, for flattening loss / multi-view homography)
    
    Returns:
        dict: {
            'render': Rendered image,
            'viewspace_points': Screen space points,
            'visibility_filter': Visibility filter,
            'radii': Radii,
            'depth': Depth map,
            'opacity': Opacity map,
            'n_touched': Touch count,
            'normal': (optional) Rendered normal map (3, H, W), unit vectors
        }
    """
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    if pc.get_xyz.shape[0] == 0:
        return None

    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        projmatrix_raw=viewpoint_camera.projection_matrix,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0], 1)
    
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # check if the covariance is isotropic
        if pc.get_scaling.shape[-1] == 1:
            scales = pc.get_scaling.repeat(1, 3)
        else:
            scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    
    # Handle dynamic deformation
    if dynamic:
        if pc.get_scaling.shape[-1] == 1:
            means3D, scales_final, rotations_final, _, _, _ = pc._deformation(
                means3D, pc._scaling.repeat(1, 3),
                pc._rotation, pc._opacity, shs, time
            )
        else:
            means3D, scales_final, rotations_final, _, _, _ = pc._deformation(
                means3D, pc._scaling,
                pc._rotation, pc._opacity, shs, time
            )
        scales = pc.scaling_activation(scales_final)
        rotations = pc.rotation_activation(rotations_final)
    
    # ========== New: Handle render_mode ==========
    # Create corresponding mask based on render_mode
    if render_mode == 'static_only':
        # Render only static Gaussians
        if hasattr(pc, 'dygs') and pc.dygs is not None and len(pc.dygs) > 0:
            # pc.dygs is a boolean mask tensor, negate it for static
            if isinstance(pc.dygs, torch.Tensor) and pc.dygs.dtype == torch.bool:
                static_mask = ~pc.dygs  # Invert the dynamic mask
            else:
                static_mask = torch.ones(means3D.shape[0], dtype=torch.bool, device=means3D.device)
                static_mask[pc.dygs] = False  # Exclude dynamic Gaussians
            
            print(f"[DEBUG] Static Gaussians (True): {static_mask.sum().item()}, Percentage: {100.0 * static_mask.sum().item() / means3D.shape[0]:.2f}%")
            
            if mask is not None:
                # If there's an existing mask, take the intersection
                mask = mask & static_mask
            else:
                mask = static_mask
            
            print(f"Rendering static Gaussians: {mask.sum().item()}/{means3D.shape[0]} Gaussians")
        else:
            print("Warning: pc.dygs does not exist or is empty, will render all Gaussians")
    
    elif render_mode == 'dynamic_only':
        # Render only dynamic Gaussians
        if hasattr(pc, 'dygs') and pc.dygs is not None and len(pc.dygs) > 0:
            # pc.dygs is a boolean mask tensor
            dynamic_mask = pc.dygs.clone() if isinstance(pc.dygs, torch.Tensor) and pc.dygs.dtype == torch.bool else torch.zeros(means3D.shape[0], dtype=torch.bool, device=means3D.device)
            
            # If pc.dygs is not boolean tensor, convert it
            if not (isinstance(pc.dygs, torch.Tensor) and pc.dygs.dtype == torch.bool):
                dynamic_mask[pc.dygs] = True
            
            # Debug: Print pc.dygs info
            print(f"[DEBUG] pc.dygs type: {type(pc.dygs)}, dtype: {pc.dygs.dtype if isinstance(pc.dygs, torch.Tensor) else 'N/A'}")
            print(f"[DEBUG] Total Gaussians: {means3D.shape[0]}, Dynamic Gaussians (True): {dynamic_mask.sum().item()}, Percentage: {100.0 * dynamic_mask.sum().item() / means3D.shape[0]:.2f}%")
            
            if mask is not None:
                mask = mask & dynamic_mask
            else:
                mask = dynamic_mask
            
            print(f"Rendering dynamic Gaussians: {mask.sum().item()}/{means3D.shape[0]} Gaussians")
            
            # If there are no dynamic Gaussians, return empty image
            if mask.sum() == 0:
                print("Warning: No dynamic Gaussians to render, returning empty image")
                H = int(viewpoint_camera.image_height)
                W = int(viewpoint_camera.image_width)
                empty_image = torch.zeros((3, H, W), device=means3D.device, requires_grad=True)
                empty_depth = torch.zeros((1, H, W), device=means3D.device, requires_grad=True)
                empty_opacity = torch.zeros((1, H, W), device=means3D.device, requires_grad=True)
                empty_radii = torch.zeros(means3D.shape[0], dtype=torch.int32, device=means3D.device)
                
                # Ensure screenspace_points has a gradient path
                screenspace_points = screenspace_points + 0
                
                return {
                    "render": empty_image,
                    "viewspace_points": screenspace_points,
                    "visibility_filter": torch.zeros_like(empty_radii, dtype=torch.bool),
                    "radii": empty_radii,
                    "depth": empty_depth,
                    "opacity": empty_opacity,
                    "n_touched": torch.zeros(1, dtype=torch.int32, device=means3D.device),
                }
        else:
            print("Warning: pc.dygs does not exist or is empty, cannot render dynamic Gaussians, returning empty image")
            H = int(viewpoint_camera.image_height)
            W = int(viewpoint_camera.image_width)
            empty_image = torch.zeros((3, H, W), device=means3D.device, requires_grad=True)
            empty_depth = torch.zeros((1, H, W), device=means3D.device, requires_grad=True)
            empty_opacity = torch.zeros((1, H, W), device=means3D.device, requires_grad=True)
            empty_radii = torch.zeros(means3D.shape[0], dtype=torch.int32, device=means3D.device)
            
            # Ensure screenspace_points has a gradient path
            screenspace_points = screenspace_points + 0
            
            return {
                "render": empty_image,
                "viewspace_points": screenspace_points,
                "visibility_filter": torch.zeros_like(empty_radii, dtype=torch.bool),
                "radii": empty_radii,
                "depth": empty_depth,
                "opacity": empty_opacity,
                "n_touched": torch.zeros(1, dtype=torch.int32, device=means3D.device),
            }
    
    elif render_mode == 'all':
        # Default mode: render all Gaussians (backward compatible)
        pass
    else:
        raise ValueError(f"Unknown render_mode: {render_mode}. Must be 'all', 'static_only' or 'dynamic_only'")
    
    # ========== Apply dynamic deformation (only for dynamic Gaussians) ==========
    if dx is not None and ds is not None and dr is not None:
        if hasattr(pc, 'dygs') and pc.dygs is not None and len(pc.dygs) > 0:
            dxyz = torch.zeros_like(means3D)
            dxyz[pc.dygs] = dx
            means3D = pc.get_xyz + dxyz
            del dxyz
            
            dscale = torch.zeros_like(scales)
            dscale[pc.dygs] = ds
            scales = scales + dscale
            del dscale
            
            drot = torch.zeros_like(rotations)
            drot[pc.dygs] = dr
            rotations = pc.get_rotation + drot
            del drot
        else:
            print("⚠️  Warning: dx/ds/dr provided but pc.dygs does not exist, skipping dynamic deformation")
    
    # PGSR-style: when return_normal=True, compute normal_precomp in caller so grad flows through scales/rotation automatically.
    normal_precomp = None
    if return_normal:
        _scales = scales if scales is not None and scales.numel() > 0 else (pc.get_scaling.repeat(1, 3) if pc.get_scaling.shape[-1] == 1 else pc.get_scaling)
        _rotations = rotations if rotations is not None and rotations.numel() > 0 else pc.get_rotation
        campos = viewpoint_camera.camera_center.squeeze()
        if campos.dim() == 0:
            campos = campos.unsqueeze(0).expand(3)
        viewmatrix = viewpoint_camera.world_view_transform
        normal_precomp = compute_shortest_axis_view(_scales, _rotations, means3D, campos, viewmatrix)
        if mask is not None:
            normal_precomp = normal_precomp[mask]
    
    # ========== Rasterize visible Gaussians to image (单 pass 可同时输出 color + normal) ==========
    if mask is not None:
        # Use mask for selective rendering
        res = rasterizer(
            means3D=means3D[mask],
            means2D=means2D[mask],
            shs=shs[mask] if shs is not None else None,
            colors_precomp=colors_precomp[mask] if colors_precomp is not None else None,
            opacities=opacity[mask],
            scales=scales[mask],
            rotations=rotations[mask],
            cov3D_precomp=cov3D_precomp[mask] if cov3D_precomp is not None else None,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
            return_normal=return_normal,
            normal_precomp=normal_precomp,
        )
    else:
        # Render all Gaussians
        res = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
            return_normal=return_normal,
            normal_precomp=normal_precomp,
        )
    if len(res) == 6:
        rendered_image, radii, depth, opacity, n_touched, normal_blend = res
    else:
        rendered_image, radii, depth, opacity, n_touched = res
        normal_blend = None

    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
        "opacity": opacity,
        "n_touched": n_touched,
    }
    if normal_blend is not None:
        # 光栅器输出为 PGSR 风格：sum(n_i * alpha * T)，此处归一化为单位法向；输出 [-1,1]，可视化用 (N+1)/2
        # 低不透明度像素的 normal 模长接近 0，归一化后不稳定，损失中需用 opacity 掩码（见 get_loss_flattening）
        n_flat = normal_blend.permute(1, 2, 0)
        n_norm = n_flat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        out["normal"] = (n_flat / n_norm).clamp(-1.0, 1.0).permute(2, 0, 1)

    return out


def render_flow(
        pc: GaussianModel,
        viewpoint_camera1,
        viewpoint_camera2,
        d_xyz1, d_xyz2,
        d_rotation1, d_scaling1,
        scaling_modifier=1.0,
        compute_cov3D_python=False,
        scale_const=None,
        d_rot_as_res=True,
        **kwargs
):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
            torch.zeros_like(
                pc.get_xyz,
                dtype=pc.get_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera1.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera1.FoVy * 0.5)

    # About Motion
    carnonical_xyz = pc.get_xyz.clone()
    xyz_at_t1 = xyz_at_t2 = carnonical_xyz.detach()  # Detach coordinates of Gaussians here
    #print("dxyz1",d_xyz1)
    dxyz1 = torch.zeros_like(xyz_at_t1)
    #print("dxyz1", dxyz1)
    dxyz1[pc.dygs] = d_xyz1
    xyz_at_t1 = xyz_at_t1 + dxyz1
    # xyz_at_t1 = xyz_at_t1 + d_xyz1
    dxyz2 = torch.zeros_like(xyz_at_t1)
    dxyz2[pc.dygs] = d_xyz2
    xyz_at_t2 = xyz_at_t2 + dxyz2
    # xyz_at_t2 = xyz_at_t2 + d_xyz2
    gaussians_homogeneous_coor_t2 = torch.cat([xyz_at_t2, torch.ones_like(xyz_at_t2[..., :1])], dim=-1)
    full_proj_transform = viewpoint_camera2.full_proj_transform if viewpoint_camera2 is not None else viewpoint_camera1.full_proj_transform
    gaussians_uvz_coor_at_cam2 = gaussians_homogeneous_coor_t2 @ full_proj_transform
    gaussians_uvz_coor_at_cam2 = gaussians_uvz_coor_at_cam2[..., :3] / (gaussians_uvz_coor_at_cam2[..., -1:] + 1e-7)

    gaussians_homogeneous_coor_t1 = torch.cat([xyz_at_t1, torch.ones_like(xyz_at_t1[..., :1])], dim=-1)
    gaussians_uvz_coor_at_cam1 = gaussians_homogeneous_coor_t1 @ viewpoint_camera1.full_proj_transform
    gaussians_uvz_coor_at_cam1 = gaussians_uvz_coor_at_cam1[..., :3] / (gaussians_uvz_coor_at_cam1[..., -1:] + 1e-7)

    flow_uvz_1to2 = gaussians_uvz_coor_at_cam2 - gaussians_uvz_coor_at_cam1

    # Rendering motion mask
    flow_uvz_1to2[..., -1:] = pc.dygs.unsqueeze(1)  # pc.motion_mask
    # print("d_xyz1 gradient mean:", d_xyz1.grad.mean().item())
    # print("d_xyz2 gradient mean:", d_xyz2.grad.mean().item())
    # print("d_rotation1 gradient variance:", d_rotation1.grad.var().item())
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera1.image_height),
        image_width=int(viewpoint_camera1.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.zeros_like(flow_uvz_1to2[0]),  # Background set as 0
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera1.world_view_transform,
        projmatrix=viewpoint_camera1.full_proj_transform,
        projmatrix_raw=viewpoint_camera1.projection_matrix,
        sh_degree=0,
        campos=viewpoint_camera1.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz + d_xyz1  # About Motion
    means3D = carnonical_xyz + dxyz1
    means2D = screenspace_points
    opacity = pc.get_opacity.clone().detach()

    if scale_const is not None:
        # If providing scale_const, directly use scale_const
        scales = torch.ones_like(pc.get_scaling) * scale_const
        if d_rot_as_res:
            rotations = pc.get_rotation + d_rotation1
        else:
            rotations = pc.get_rotation if type(d_rotation1) is float else quaternion_multiply(d_rotation1,
                                                                                               pc.get_rotation)
        cov3D_precomp = None
    else:
        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier,
                                              d_rotation=None if type(d_rotation1) is float else d_rotation1)
        else:
            dscale = torch.zeros_like(pc.get_scaling)
            dscale[pc.dygs] = d_scaling1
            scales = pc.get_scaling.clone().detach() + dscale
            del dscale
            # scales = pc.get_scaling + d_scaling1
            if d_rot_as_res:
                drot = torch.zeros_like(pc.get_rotation)
                drot[pc.dygs] = d_rotation1
                rotations = pc.get_rotation.clone().detach() + drot
                del drot
                # rotations = pc.get_rotation + d_rotation1
            else:
                rotations = pc.get_rotation if type(d_rotation1) is float else quaternion_multiply(d_rotation1,
                                                                                                   pc.get_rotation)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, rendered_depth, rendered_alpha, n_touched= rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=flow_uvz_1to2,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "depth": rendered_depth,
        "alpha": rendered_alpha,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }

def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)
def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)
def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)
def get_dynamic_mask(
        viewpoint_camera,
        pc: GaussianModel,
        pipe,
        override_color=None,
        dynamic=True,
):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    if pc.get_xyz.shape[0] == 0:
        return None

    means3D = pc.get_xyz.clone().detach()

    time = torch.tensor(viewpoint_camera.time - 1).to(means3D.device).repeat(means3D.shape[0], 1)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if dynamic:
        if pc.get_scaling.shape[-1] == 1:
            _, _, _, dx, ds, dr = pc._deformation(means3D, pc._scaling.repeat(1, 3).clone().detach(),
                                                  pc._rotation.clone().detach(), pc._opacity.clone().detach(),
                                                  shs.clone().detach(), time)
        else:
            _, _, _, dx, ds, dr = pc._deformation(means3D, pc._scaling,
                                                  pc._rotation, pc._opacity, shs, time)
        print(torch.norm(dx, dim=1).mean(), torch.norm(ds, dim=1).mean(), torch.norm(dr, dim=1).mean())
        print(torch.norm(dx, dim=1).max(), torch.norm(ds, dim=1).max(), torch.norm(dr, dim=1).max())
        position_mask = (torch.norm(dx, dim=1) < 1)
        scale_mask = (torch.norm(ds, dim=1) < 2)
        direction_mask = (torch.norm(dr, dim=1) < 1)

        static_mask = position_mask & scale_mask & direction_mask
        return static_mask


