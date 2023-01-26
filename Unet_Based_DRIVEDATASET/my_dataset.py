#TODO: Load the Drive dataset
import os
from PIL import  Image
import numpy as np
from torch.utils.data import Dataset

#TODO:Inherit Dataset

class DriveDataset(Dataset):
    def __init__(self, root_dir:str,train:bool, transforms=None):
        super(DriveDataset, self).__init__()
        self.flag = 'training' if train else 'test'
        data_root = os.path.join(root_dir,'DRIVE','datasets',self.flag)
        # write the assert to avoid problems with do not find the data path
        assert os.path.exists(data_root), f'The path {data_root} does not exist'
        
        self.transforms = transforms
        img_names  = [i for i in os.listdir(os.path.join(data_root,'images')) if i.endswith('.tif')]
        '''
        for i in img_names:
            print(i[0:3])
        '''
        self.img_list = [os.path.join(data_root,'images',i) for i in img_names]
        self.manual = [os.path.join(data_root,'1st_manual',i[0:3]+'manual1.gif') for i in img_names]
        #/home/zhuangzhi_gao/Desktop/Zhuangzhi_Gao_Daily_Life/U_net_Based_Retinal_Photography/DRIVE/datasets/training
        #check if the files exist
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f'file{i} does not exists.')
        
        self.roi_mask = [os.path.join(data_root, 'mask',i.split('_')[0]+f'_{self.flag}_mask.gif')
        for i in img_names]

        #check if the files exist
        for i in self.roi_mask:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f'file{i} does not exists.')
    
    def __getitem__(self, index):
        img = Image.open(self.img_list[index]).convert('RGB')
        manual = Image.open(self.manual[index]).convert('L')
        manual = np.array(manual)/255 # Foreground pixel value is 255 and background pixel value is 0
        roi_mask = Image.open(self.roi_mask[index]).convert('L')
        roi_mask = 255 - np.array(roi_mask)# The interesting area is 0 the ununteresting area is 255
        #The foreground pixel value is 1
        #The background pixel value is 0
        #the ununteresting area is 255
        mask = np.clip(manual+roi_mask,a_min =0,a_max=255)

        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img,mask = self.transforms(img,mask)
        return img,mask
    def __len__(self):
        return len(self.img_list)
    

    @staticmethod
    def collate_fn(batch):
        images,targets = list(zip(*batch))
        batch_imgs = cat_list(images,fill_values=0)
        batch_targets = cat_list(targets,fill_values=0)
        return batch_imgs,batch_targets
    
def cat_list(images,fill_values=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_values)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


