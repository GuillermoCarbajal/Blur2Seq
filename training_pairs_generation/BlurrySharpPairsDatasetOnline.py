
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np

class BlurrySharpPairDatasetOnline(Dataset):
    def __init__(self, sharp_folder, positions_folder, 
                 transform=None, seed=None,  n_positions=25, rotate=False, 
                 random_focal_length=False, augment_illumination=False, jitter_illumination=1.0, random_focal_length_octaves=3.0, 
                 gamma_factor=None, exponential_factor=None, Kohler_intrinsics=False,
                 crop_size=256, image_size=-1, augment_trajectories=False, roll_shift=False):
        """
        Args:
             split_file: Path to the split file
             root_dir: Directory with all the images
             transform: Optional transform to be appeared on a sample
        """
        self.sharp_image_files = os.listdir(sharp_folder)
        #self.sharp_image_files = self.sharp_image_files[0:2000:200]
        self.positions_files = os.listdir(positions_folder)
        self.sharp_image_files.sort()
        self.positions_files.sort()
        

        #self.blur_image_files = self.blur_image_files[:5]
        #self.sharp_image_files = self.sharp_image_files[:100]

        self.transform = transform
        self.seed = seed
        self.n_positions = n_positions
        self.rotate = rotate
        self.sharp_folder = sharp_folder
        self.positions_folder = positions_folder
        self.random_focal_length=random_focal_length
        self.augment_illumination = augment_illumination
        self.jitter_illumination = jitter_illumination
        self.Kohler_intrinsics=Kohler_intrinsics
        self.augment_trajectories = augment_trajectories
        self.random_focal_length_octaves = random_focal_length_octaves
        self.crop_size=crop_size
        self.image_size=image_size
        self.gamma_factor=gamma_factor
        self.exponential_factor=exponential_factor
        self.roll_shifts=np.zeros(len(self.positions_files))
        if roll_shift:
            self.roll_shifts=(-1+2*np.random.rand(len(self.positions_files)))*np.pi/180




    def __len__(self):
        return len(self.sharp_image_files)  
    
    
    def rgb2lin_exp(self, img, a=7,b=1):    
        log_x = np.log( -img + 1e-9 + 1   )/b
        photons = -log_x /a
        return photons
    
    def __getitem__(self, idx):
  
        #idx = np.random.randint(3) #np.random.randint(3)
        sharp_image_name = self.sharp_image_files[idx]
        color=False  
        minimum_size=False
        while not color or not minimum_size:
            sharp_image_name = self.sharp_image_files[idx]
            sharp_image = Image.open(os.path.join(self.sharp_folder,sharp_image_name))
            sharp_image = np.array(sharp_image)
            color = len(sharp_image.shape)==3
            H0, W0 = sharp_image.shape[:2]
            min_size = np.min([H0,W0])
            minimum_size = min_size>(self.crop_size+64) if self.image_size<0 else min_size>self.image_size
            if not color or not minimum_size:
                idx = np.random.randint(len(self.sharp_image_files)) 
            elif self.image_size > 0:
                i0 = np.random.randint(H0-self.image_size)
                j0 = np.random.randint(W0-self.image_size)
                sharp_image = sharp_image[i0:i0+self.image_size,j0:j0+self.image_size,:]

        pos_idx = np.random.randint(len(self.positions_files))     
        camera_file = self.positions_files[pos_idx]


        camera_positions = np.loadtxt(os.path.join(self.positions_folder,camera_file), delimiter=',')
        camera_positions = camera_positions[:,3:] # only angles
        camera_positions -= camera_positions.mean(axis=0)
        camera_positions[:,2]+=self.roll_shifts[pos_idx]

        n_pos_in = len(camera_positions)
        original_indices = np.linspace(0,n_pos_in-1, n_pos_in)
        interpolated_indices = np.linspace(0,n_pos_in-1, self.n_positions)
        interpolated_positions = np.zeros((self.n_positions, 3))
        for p in range(camera_positions.shape[1]):
            interpolated_positions[:,p]=np.interp(interpolated_indices, original_indices, camera_positions[:,p])
        
        #sharp_image = sharp_image[:,::-1,:] if np.random.rand() > 0.5 else sharp_image    
        sharp_image = torch.from_numpy(sharp_image/255.0).permute(2,0,1).float()
        sharp_image = sharp_image**self.gamma_factor if self.gamma_factor is not None else sharp_image
        sharp_image = self.rgb2lin_exp(sharp_image, self.exponential_factor) if self.exponential_factor is not None else sharp_image


        if self.augment_illumination and np.random.rand()>0.25:
            sharp_image*=(1+self.jitter_illumination*np.random.rand())

        if self.rotate:
            orientation = [1,2] if np.random.rand() > 0.5 else [2, 1]
            times = np.random.choice(4)

            sharp_image = torch.rot90(sharp_image,dims=orientation, k=times).float()

            if orientation==[2,1]:
                rotation_matrix=np.array([[np.cos(90*times*np.pi/180), -np.sin(90*times*np.pi/180), 0],
                                          [np.sin(90*times*np.pi/180),  np.cos(90*times*np.pi/180), 0],
                                          [0,0,1]])
            elif orientation==[1,2]:
                rotation_matrix=np.array([[np.cos(-90*times*np.pi/180), -np.sin(-90*times*np.pi/180), 0],
                                          [np.sin(-90*times*np.pi/180),  np.cos(-90*times*np.pi/180), 0],
                                          [0,0,1]])
            interpolated_positions=interpolated_positions@rotation_matrix.T
        

        C, H, W = sharp_image.shape

       
        f0=np.max([H0,W0])   # largo focal de la original
        if self.random_focal_length_octaves == -1:
            f = (1+np.random.rand()*5)*f0
        elif self.random_focal_length_octaves == -2:
            f = (1+np.random.rand()*2)*f0            
        else:
            f = f0/2*2**(self.random_focal_length_octaves*np.random.rand()) if self.random_focal_length  else f0 # (1+np.random.rand()*3)*f0
        pi = H/ 2
        pj = W/ 2
        intrinsics = np.array([[f, 0, pj], [0, f, pi], [0, 0, 1]])

        if self.Kohler_intrinsics:
            #intrinsics = np.array([[3900, 0, 400], [0, 3900, 400], [0, 0, 1]])
            intrinsics = np.array([[3900, 0, pj], [0, 3900, pi], [0, 0, 1]])
            scale = 2*(0.5+4*np.random.rand())
            interpolated_positions = interpolated_positions/(scale)  # porque las rotaciones son muy grandes para este intrinsics
            #intrinsics = np.array([[3900*800/f0, 0, 400*pj/400], [0, 3900*800/f0, 400*pi/400], [0, 0, 1]])  # Kohler intrinsics
        if self.augment_trajectories and np.random.rand()>0.5:
            if f> 1500:
                interpolated_positions = interpolated_positions/2   # para que en los largos focales altos no todas las imágenes sean muy borrosas
            elif f<1500:  # multiplico por 4 el f y divido entre cuatro las posiciones
                intrinsics = np.array([[4*f, 0, pj], [0, 4*f, pi], [0, 0, 1]])
                interpolated_positions = interpolated_positions/4


        intrinsics = torch.from_numpy(intrinsics).float()

        return {'sharp_image': sharp_image, 'positions': torch.FloatTensor(interpolated_positions), 'intrinsics': torch.FloatTensor(intrinsics)}

