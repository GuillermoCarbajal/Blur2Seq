import torchvision
import numpy as np
import torch
from PIL import Image
from skimage.io import imread, imsave
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from matplotlib import pyplot as plt



# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
	image_numpy = image_tensor.cpu().float().numpy()
	image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 0.5) * 255.0
	return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
	image_pil = None
	if image_numpy.shape[2] == 1:
		image_numpy = np.reshape(image_numpy, (image_numpy.shape[0],image_numpy.shape[1]))
		image_pil = Image.fromarray(image_numpy, 'L')
	else:
		image_pil = Image.fromarray(image_numpy)
	image_pil.save(image_path)


def show_positions_found(pos, f=None, img_name=None):
    fig = plt.figure(figsize=(14,7))
    ax1 = fig.add_subplot(1,2,1,projection='3d')
    ax1.plot(pos[:,0], pos[:,1], pos[:,2], '*-', label='positions found')
    ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], '*-', label='gt_positions')
    ax1.plot([0],[0],[0],'*r')
    ax1.plot(pos[0, 0], pos[0, 1], pos[0, 2], '*g', label='start point')
    ax1.plot(pos[0, 0], pos[0, 1], pos[0, 2], '*g', label='start point')
    ax1.plot(pos[-1, 0], pos[-1, 1], pos[-1, 2], '*m', label='end point')
    ax1.plot(pos[-1, 0], pos[-1, 1], pos[-1, 2], '*m', label='end point')
    ax1.set_xlim(-0.01,0.01)
    ax1.set_ylim(-0.01,0.01)
    ax1.set_zlim(-0.01,0.01)
    ax1.set_xlabel('thetaX')
    ax1.set_ylabel('thetaY')
    ax1.set_zlabel('thetaZ')
    ax1.view_init(elev=10., azim=15)

    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(pos[:,0], label='rot_x')
    ax2.plot(pos[:,1], label='rot_y')
    ax2.plot(pos[:,2], label='rot_z')
    plt.legend()

    if f is not None:
        ax1.set_title(f'f={f:.01f}')
        ax2.set_title('Trajectories')
    
    if img_name is not None:
        fig.savefig(img_name)
        
    plt.close()

    return    

def sort_positions(found_positions):
    dist_matrix = torch.cdist(found_positions, found_positions, p=2)
    suma_distancias = torch.sum(dist_matrix,axis=0)
    ind_extremo = torch.argmax(suma_distancias)
    order = torch.argsort(dist_matrix[ind_extremo], descending=True)
    return order
    
def save_video(frames, output_file):
        '''
        frames: list of frames
        '''
        cv_images=[]
        B,C,H,W = frames[0].shape
        position = (W-100, 50)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        color = (0, 255, 0)
        thickness = 2
        #blurry_to_draw = cv2.cvtColor((255*blurry_image).astype(np.uint8), cv2.COLOR_RGB2BGR)
        #blurry_to_draw = cv2.putText(cv2.UMat(blurry_to_draw), 'Blurry', position, font, font_scale, color, thickness)
        #cv_images.append(blurry_to_draw)
        for n in range(len(frames)):
            sharp_n = tensor2im(torch.clamp(frames[n][0].detach(),0,1) - 0.5)
            sharp_n_draw = cv2.putText(cv2.UMat(sharp_n), str(n), position, font, font_scale, color, thickness)
            #imgs.append(Image.fromarray(np.uint8(sharp_n)).convert("P",palette=Image.ADAPTIVE))
            cv_images.append(cv2.cvtColor(sharp_n_draw, cv2.COLOR_RGB2BGR))   
        
        video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'MJPG'), 25, (W,H))
        for image in cv_images:
            video.write(image)
        video.release()

        del cv_images    
