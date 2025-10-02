import torch
import kornia
import numpy as np
from kornia.utils import create_meshgrid
from kornia.geometry.transform import warp_grid
from kornia.geometry.conversions import rotation_matrix_to_angle_axis
from torchvision.ops import deform_conv2d
from skimage.color import rgb2gray
from skimage.io import imsave


def initial_offset_5x5():
    
    p = torch.Tensor(25,2,1,1)   #  P, 2, 1, 1

    p[0,0]=-2; p[0,1]=-2
    p[1,0]=-2; p[1,1]=-1
    p[2,0]=-2; p[2,1]=0
    p[3,0]=-2; p[3,1]=1
    p[4,0]=-2; p[4,1]=2

    p[5,0]=-1; p[5,1]=-2
    p[6,0]=-1; p[6,1]=-1
    p[7,0]=-1; p[7,1]=0
    p[8,0]=-1; p[8,1]=1
    p[9,0]=-1; p[9,1]=2


    p[10,0]=0; p[10,1]=-2
    p[11,0]=0; p[11,1]=-1
    p[12,0]=0; p[12,1]=0
    p[13,0]=0; p[13,1]=1
    p[14,0]=0; p[14,1]=2

    p[15,0]=1; p[15,1]=-2
    p[16,0]=1; p[16,1]=-1
    p[17,0]=1; p[17,1]=0
    p[18,0]=1; p[18,1]=1
    p[19,0]=1; p[19,1]=2

    p[20,0]=2; p[20,1]=-2
    p[21,0]=2; p[21,1]=-1
    p[22,0]=2; p[22,1]=0
    p[23,0]=2; p[23,1]=1
    p[24,0]=2; p[24,1]=2

    return p
    
def kornia_compute_rotation_matrix(angle_axis, eps=1e-6):
        
        angle_axis_clone = torch.clone(angle_axis)
        _angle_axis = torch.unsqueeze(angle_axis_clone, dim=1)
        theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
        theta2 = torch.squeeze(theta2, dim=1)

        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        theta = torch.sqrt(theta2)
        wxyz = angle_axis_clone / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.reshape(1, 3, 3)   
        
def compute_intrinsics(W,H, border_size=0):
    '''
    After the blurry/sharp pairs were generated, border_size pixels were removed on each side.
    As we are working with a crop, the focal length must be modified accordingly
    '''
    f = np.max([W+2*border_size, H+2*border_size])
    pi = H / 2
    pj = W / 2
    A = torch.Tensor([[f, 0, pj], [0, f, pi], [0, 0, 1]])
    return A
    
def compute_projection_matrix(camera_position):
    '''
    camera position: (3) array with rotation angles
    '''

    #P = kornia.geometry.angle_axis_to_rotation_matrix(camera_position[None, :])
    P = kornia_compute_rotation_matrix(camera_position[None])


    return P

def reblur_offsets(sharp, offsets, forward=True, mask=None) :

    offsets = torch.clone(offsets)
    B, C, H, W = sharp.shape
    B, n_positions, D, _, _ = offsets.shape
    n_positions = n_positions*D//2
    kh, kw = int(np.sqrt(n_positions)), int(np.sqrt(n_positions))
    #kh, kw = 5,5   # we have 25 offsets
    
    if kw==5:
        p = initial_offset_5x5().to(offsets.device)
    elif kw==7:
        p = initial_offset_7x7().to(offsets.device)
    else:
        print('offsets could not be initialized, please check dimensions. offsets input must be n_positions x 2 x H x W')
    

   
    weight = torch.ones((C,1,kh, kw)).to(offsets.device)/(kh*kw)
    #weight = torch.zeros((C,1,kh, kw)).cuda()
    #weight[:,0,4,4]=1
    #mask = torch.ones(1, kh * kw, H, W).cuda()
    if not forward:
        offsets*=-1

    offsets_dc =   offsets.reshape(B,2*n_positions,H,W) - p.reshape(1,2*n_positions,1,1) # 

    pad= int(torch.abs(offsets_dc).max())+1
    pad2d = torch.nn.ReflectionPad2d(pad)
    sharp_gt_padded = pad2d(sharp)
    offsets_padded = pad2d(offsets_dc)
 
    #offsets_padded = offsets_padded.reshape(B,n_positions,2,H+2*pad,W+2*pad)
    #offsets_padded[:,:,0,0:pad,:]  = -offsets_padded[:,:,0,0:pad,:]    # filp arriba
    #offsets_padded[:,:,0,H+pad:,:] = -offsets_padded[:,:,0,H+pad:,:]     # filp abajo
    #offsets_padded[:,:,1,:,0:pad]  = -offsets_padded[:,:,1,:,0:pad]     # filp izquierda
    #offsets_padded[:,:,1,:,H+pad:] = -offsets_padded[:,:,1,:,H+pad:]   # filp derecha
    #offsets_padded = offsets_padded.reshape(B,n_positions*2,H+2*pad,W+2*pad)

    offsets_padded = offsets_padded.reshape(B,n_positions,2,H+2*pad,W+2*pad)
    offsets_padded[:,:,1,0:pad,:]  = - offsets_padded[:,:,1,0:pad,:]    # reflextion arriba
    offsets_padded[:,:,1,H+pad:,:] = -offsets_padded[:,:,1,H+pad:,:] #-offsets_padded[:,:,0,Hs+pad:,:]     # reflextion abajo
    offsets_padded[:,:,0,:,0:pad]  = -offsets_padded[:,:,0,:,0:pad]  #-offsets_padded[:,:,1,:,0:pad]     # reflextion izquierda
    offsets_padded[:,:,0,:,W+pad:] = -offsets_padded[:,:,0,:,W+pad:] #-offsets_padded[:,:,1,:,Hs+pad:]   # reflextion derecha
    offsets_padded = offsets_padded.reshape(B,n_positions*2,H+2*pad,W+2*pad)



    #reblurred = deform_conv2d(sharp_gt_padded, offsets_padded, weight, padding=kw//2, mask=mask)
    reblurred = deform_conv2d(sharp_gt_padded, offsets_padded, weight, padding=kw//2, mask=mask)
    reblurred = reblurred[:,:,pad:-pad, pad:-pad]
    
    return reblurred
    
def get_offsets_from_positions(blurry_shape, positions, intrinsics, steps=[1,1], adjunct_operator_offsets=False):
    
    '''
    Method invoked from lineal assignement example

    blurry_shape: (4)
    positions: (B,N,3)
    intrinsics: (B,3,3)
    steps: separation between computed offsets
    adjunct_operator_offsets: whether to compute offsets associated with the inverse homography 
    '''
    B, C, H, W = blurry_shape
    N = positions.shape[1]
    positions = positions.reshape(B*N,3)
    n_positions = B*N
    src_homo_dst = torch.zeros(n_positions,3,3).to(positions.device)
    dst_homo_src = torch.zeros(n_positions,3,3).to(positions.device) 
    for n in range(n_positions):
        positions_input = positions[n,3:] if positions.shape[1]==6  else positions[n]
        b = n//N
        if adjunct_operator_offsets:
            # when we use the adjoint blur operator offsets 
            dst_homo_src_n = compute_homography_from_position(positions_input, intrinsics[b], normalize=False,inverse=True)
        else:
            # by default 
            dst_homo_src_n = compute_homography_from_position(positions_input, intrinsics[b], normalize=False) 

        # print('dst_homo_src ', dst_homo_src)
        src_homo_dst_n: torch.Tensor = torch.inverse(dst_homo_src_n)
        src_homo_dst[n]+=src_homo_dst_n[0,0]
        dst_homo_src[n]+=dst_homo_src_n[0,0]
        # print('iter %d:' %i, 'dst_homo_src: ', dst_homo_src)
        # print('src_homo_dst', src_homo_dst)

    # create base grid to compute the flow
    
    grid: torch.Tensor = create_meshgrid(H, W, normalized_coordinates=False).to(positions.device)  # 1, H, W, 2
    grid = grid[:,::steps[0],::steps[1],:]
    warped_grid = warp_grid(grid, src_homo_dst[:,None,:,:])  #  P, H, W, 2

    offsets = warped_grid - grid

    offsets= offsets.permute(0,3,1,2)    #  P, 2, H, W
    # offsets changed from x,y to y,x
    offsets = torch.flip(offsets,dims=[1])

    offsets = offsets.reshape(B,N,2,offsets.shape[2],offsets.shape[3])

    return offsets
    
def compute_homography_from_position(camera_position, intrinsics, inverse=False, normalize=True):
    '''
    camera position: (3) array with rotation angles
    '''
    
    N = 2 * intrinsics[0, 2]
    M = 2 * intrinsics[1, 2]
    P = compute_projection_matrix(camera_position) 

    dst_homo_src = intrinsics @ P @ torch.inverse(intrinsics) 

    if inverse:
        dst_homo_src = torch.inverse(dst_homo_src)

    dst_homo_src = torch.unsqueeze(dst_homo_src, dim=0)

    if normalize:
        dst_homo_src = kornia.geometry.conversions.normalize_homography(dst_homo_src, (M, N), (M, N))

    return dst_homo_src    

def generate_video(sharp_image, camera_positions, intrinsics, forward = True):
    '''
    sharp_image: BxCxHxW
    camera_positions: BxPx3
    intrinsics: 3x3
    '''
    H = sharp_image.size(2)
    W = sharp_image.size(3)
    reblured_image = torch.zeros_like(sharp_image).to(sharp_image.device)
    n_positions = camera_positions.shape[1]
    warper = kornia.geometry.HomographyWarper(H, W, padding_mode='reflection')
    frames=[]
    for n in range(n_positions):
        if forward:
            dst_homo_src_n = compute_homography_from_position(camera_positions[0, n, :], intrinsics)
        else:
            dst_homo_src_n = compute_homography_from_position(camera_positions[0, n, :], intrinsics, inverse=True)

        # dst_homo_src = torch.unsqueeze(intrinsics @ camera_positions, dim=0)
        # dst_homo_src  = camera_model()

        # print('dst_homo_src ', dst_homo_src)

        src_homo_dst_n: torch.Tensor = torch.inverse(dst_homo_src_n)


        # print('iter %d:' %i, 'dst_homo_src: ', dst_homo_src)
        # print('src_homo_dst', src_homo_dst)
        img_src_to_dst_n = warper(sharp_image, src_homo_dst_n)
        frames.append(img_src_to_dst_n)
        
        reblured_image += img_src_to_dst_n / n_positions

    return frames, reblured_image
    
def save_kernels_from_offsets(offsets,  output_name):
    offsets = offsets.clone()
    n_positions ,D , H, W = offsets.shape
    n_positions = n_positions*D//2
    kh, kw = int(np.sqrt(n_positions)), int(np.sqrt(n_positions))
    #kh, kw = 5, 5   # we have 25 offsets
    
    if kh==5:
        off0 = initial_offset_5x5()
        p = torch.reshape(off0,(1,2*kh*kw,1,1)).to(offsets.device)
    elif kh==7:
        off0=initial_offset_7x7()
        p = torch.reshape(off0,(1,2*kh*kw,1,1)).to(offsets.device)
    else:
        print('offsets could not be initialized, check dimensions. Input must be n_positions x 2 x H x W')
        return
    
    C=3
    weight = torch.ones((C,1,kh, kw)).to(offsets.device)/(kh*kw)

    offsets = torch.reshape(offsets,(1,2*kh*kw,H,W))
    a = torch.zeros(1,1,H,W).to(offsets.device)
    a[:,:,32::65,32::65] = 1.0
    kernel_field = deform_conv2d(a, offsets-p, weight, padding=weight.shape[-1]//2)
    kernel_field = kernel_field[0].permute(1,2,0).detach().cpu().numpy()
    kernel_image = (kernel_field-kernel_field.min())/(kernel_field.max()-kernel_field.min())
    imsave(output_name, (255*kernel_image).astype(np.uint8))

    
def show_kernels_from_offsets_on_blurry_image(blurry, offsets,  output_name):

    '''
     Draw and save CONVOLUTION kernels in the blurry image.
     Notice that computed kernels are CORRELATION kernels, therefore are flipped.
    :param blurry_image: Tensor (channels,M,N)
    :param kernels_image: Tensor (M,N)
    :return:
    '''
    C = blurry.size(0)
    M = blurry.size(1)
    N = blurry.size(2)

    blurry_image = blurry.cpu().numpy()


    offsets = offsets.clone()
    n_positions ,D , H, W = offsets.shape
    n_positions = n_positions*D//2
    kh, kw = int(np.sqrt(n_positions)), int(np.sqrt(n_positions))
    #kh, kw = 5, 5   # we have 25 offsets
    
    if kh==5:
        off0 = initial_offset_5x5()
        p = torch.reshape(off0,(1,2*kh*kw,1,1)).to(offsets.device)
    elif kh==7:
        off0=initial_offset_7x7()
        p = torch.reshape(off0,(1,2*kh*kw,1,1)).to(offsets.device)
    else:
        print('offsets could not be initialized, check dimensions. Input must be n_positions x 2 x H x W')
        return
    
    C=3
    weight = torch.ones((C,1,kh, kw)).to(offsets.device)/(kh*kw)

    offsets = torch.reshape(offsets,(1,2*kh*kw,H,W))
    a = torch.zeros(1,1,H,W).to(offsets.device)
    step = 33 if torch.max(abs(offsets)) < 33 else 65
    #a[:,:,32::65,32::65] = 1.0
    a[:,:,33::step,33::step] = 1.0
    kernel_field = deform_conv2d(a, offsets-p, weight, padding=weight.shape[-1]//2)
    kernel_field = kernel_field[0].permute(1,2,0).detach().cpu().numpy()
    kernels_image = (kernel_field-kernel_field.min())/(kernel_field.max()-kernel_field.min())


    kernels_image = kernels_image.transpose(2,0,1)
    grid_to_draw = 0.4*1 + 0.6*rgb2gray(blurry_image.transpose(1,2,0)).copy()
    grid_to_draw = np.repeat(grid_to_draw[None,:,:], 3, axis=0)
    
    #grid_to_draw[0] = 0.6 * kernels_image[0] + (1- kernels_image[0]) * grid_to_draw[0]
    grid_to_draw[0] = 25*kernels_image[0] + (1- kernels_image[0]) * grid_to_draw[0]
    grid_to_draw[1:] = (1- kernels_image[1:]) * grid_to_draw[1:]


    grid_to_draw = np.clip(grid_to_draw, 0, 1)
    imsave(output_name, (255*grid_to_draw.transpose((1, 2, 0))).astype(np.uint8))    
    #imsave(output_name, (255*kernel_image).astype(np.uint8))    
