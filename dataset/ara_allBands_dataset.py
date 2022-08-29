import numpy as np
import os
import torch
from torch.utils.data import Dataset

# path = "../data/earthnet/augment/"
# list_IDs = os.listdir(path)
# labels = [i for i in range(len(list_IDs))]

    
class TriDataset(Dataset):

    def load(self, path):
        data = np.load(path, allow_pickle=True).item()
        return data

    def get_feat_label(self, triplet, mask):
        # mask initially in 128x128
        landsat = triplet["landsat"]
        target = triplet["target"]
        sentinel = triplet["sentinel"]
        
        mask3 = mask.repeat(3, axis=0).repeat(3, axis=1)
#         mask = np.stack((mask, mask, mask), axis=2)
        mask3 = np.stack((mask3, mask3, mask3), axis=2)
        
        target = np.nan_to_num(target)
        
#         target[:, :, 0, :] = (target[:, :, 0, :] - (-0.1385498046875)) / (2.375 - (-0.1385498046875)) - 0.5
#         target[:, :, 1, :] = (target[:, :, 1, :] - (-0.165771484375)) / (2.14453125 - (-0.165771484375) ) - 0.5
#         target[:, :, 2, :] = (target[:, :, 2, :] - (-0.1611328125)) / (2.021484375 - (-0.1611328125)) - 0.5
#         target[:, :, 3, :] = (target[:, :, 3, :] - (-0.1553955078125)) / (1.8251953125 - (-0.1553955078125)) - 0.5

#         target = target.reshape((384, 384, -1), order = "F")
        target = np.transpose(target, (2, 0, 1))
        filling1 = np.zeros((128, 128, 3))
        filling3 = np.zeros((384, 384, 3))
#         filling[:, :, :, 1] = mask[:, :, :, 0]
#         mask = filling.reshape((128, 128, -1), order = "F")
#         mask = np.transpose(mask, (2, 0, 1))
        
        cloudy = target * (1. - mask3)

        return landsat, sentinel, cloudy, target, mask3

    def __init__(self, path):
        # super().__init__(batch_size = batch_size, shuffle = shuffle)
        self.path = path
        self.list_IDs = os.listdir(self.path)
        self.labels = [i for i in range(len(self.list_IDs))]

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        landsat, sentinel, cloudy, target, mask3 = self.get_feat_label(self.load_all(self.path + ID))
        # Load data and get label
        # X = torch.load('data/' + ID + '.pt')
        # y = self.labels[ID]

        sample = {"landsat": landsat, "sentinel": sentinel, "cloudy": cloudy, "target": target, "mask": mask3, "id": ID}
        return sample
    

# class TriImageDataset_CF(Dataset):

#     def load_all(self, path):
#         data = np.load(path)
#         return data

#     def get_feat_label(self, cube):
#         cube_highresdynamic = cube.f.highresdynamic
#         mid = cube_highresdynamic.shape[3] // 2
#         # target = cube_highresdynamic[:, :, 0:7, mid:mid + 1]
#         mask = cube_highresdynamic[:, :, -1:, 0]
#         if self.trans_cloud:
#             if self.loc == "TL":
#                 row_start = 0
#                 col_start = 0
#             elif self.loc == "MV": 
#                 row_start = self.rowi * self.cloud_stride
#                 col_start = self.coli * self.cloud_stride
#             elif self.loc == "C":
#                 row_start = 64 - self.cloud_l//2
#                 col_start = 64 - self.cloud_l//2
            
#             row_end = row_start + self.cloud_l
#             col_end = col_start + self.cloud_l
#             mask = np.zeros_like(mask)
#             mask[row_start:row_end, col_start:col_end,:] = 1
#             #print(np.sum(mask))
#         mask = np.stack((mask, mask, mask, mask), axis=2)
#         # target = np.concatenate((cube_highresdynamic[:, :, 0:4, 0:mid], cube_highresdynamic[:, :, 0:4, mid:mid + 1], cube_highresdynamic[:, :, 0:4, mid + 1:]),
#         #                        axis=3)
#         target = cube_highresdynamic[:, :, 0:4, :]
#         target = np.nan_to_num(target)
# #         target[:, :, 0, :] = (target[:, :, 0, :] - (-0.1385498046875)) / (2.375 - (-0.1385498046875)) - 0.5
# #         target[:, :, 1, :] = (target[:, :, 1, :] - (-0.165771484375)) / (2.14453125 - (-0.165771484375) ) - 0.5
# #         target[:, :, 2, :] = (target[:, :, 2, :] - (-0.1611328125)) / (2.021484375 - (-0.1611328125)) - 0.5
# #         target[:, :, 3, :] = (target[:, :, 3, :] - (-0.1553955078125)) / (1.8251953125 - (-0.1553955078125)) - 0.5
#         target = target.reshape((128, 128, -1), order = "F")
#         target = np.transpose(target, (2, 0, 1))
#         filling = np.zeros((128, 128, 4, 3))
#         filling[:, :, :, 1] = mask[:, :, :, 0]
#         mask = filling.reshape((128, 128, -1), order = "F")
#         mask = np.transpose(mask, (2, 0, 1))
#         features = target * (1. - mask)
        
#         return features, target, mask

#     def __init__(self, path, trans_cloud = None, cloud_l = 32, cloud_stride = 32, rowi = 0, coli = 0, loc = "TL"):
#         # super().__init__(batch_size = batch_size, shuffle = shuffle)
#         self.path = path
#         self.list_IDs = os.listdir(self.path)
#         self.labels = [i for i in range(len(self.list_IDs))]
#         self.trans_cloud = trans_cloud
#         self.cloud_l = cloud_l
#         self.cloud_stride = cloud_stride
#         self.rowi = rowi
#         self.coli = coli
#         self.loc = loc
        
#     def __len__(self):
#         return len(self.list_IDs)

#     def __getitem__(self, index):
#         ID = self.list_IDs[index]

#         features, target, mask = self.get_feat_label(self.load_all(self.path + ID))
#         # Load data and get label
#         # X = torch.load('data/' + ID + '.pt')
#         # y = self.labels[ID]

#         sample = {"features": features, "target": target, "mask": mask, "id": ID}
#         return sample
    
# class TriImageDataset_CF_DS(Dataset):

#     def load_all(self, path):
#         data = np.load(path)
#         return data

#     def get_feat_label(self, cube):
#         cube_highresdynamic = cube.f.highresdynamic
#         mid = cube_highresdynamic.shape[3] // 2
#         # target = cube_highresdynamic[:, :, 0:7, mid:mid + 1]
#         mask = cube_highresdynamic[:, :, -1:, 0]
#         mask = np.stack((mask, mask, mask, mask), axis=2)
#         # target = np.concatenate((cube_highresdynamic[:, :, 0:4, 0:mid], cube_highresdynamic[:, :, 0:4, mid:mid + 1], cube_highresdynamic[:, :, 0:4, mid + 1:]),
#         #                        axis=3)
#         target = cube_highresdynamic[:, :, 0:4, :]
#         target = np.nan_to_num(target)
# #         target[:, :, 0, :] = (target[:, :, 0, :] - (-0.1385498046875)) / (2.375 - (-0.1385498046875)) - 0.5
# #         target[:, :, 1, :] = (target[:, :, 1, :] - (-0.165771484375)) / (2.14453125 - (-0.165771484375) ) - 0.5
# #         target[:, :, 2, :] = (target[:, :, 2, :] - (-0.1611328125)) / (2.021484375 - (-0.1611328125)) - 0.5
# #         target[:, :, 3, :] = (target[:, :, 3, :] - (-0.1553955078125)) / (1.8251953125 - (-0.1553955078125)) - 0.5
#         target = target.reshape((128, 128, -1), order = "F")
#         target = np.transpose(target, (2, 0, 1))
#         filling = np.zeros((128, 128, 4, 3))
#         filling[:, :, :, 1] = mask[:, :, :, 0]
#         mask = filling.reshape((128, 128, -1), order = "F")
#         mask = np.transpose(mask, (2, 0, 1))
#         features = target * (1. - mask)
        
# #         prev = features[0:4, ::2, ::2]
# #         mid = features[4:8, :, :]
# #         after = features[8:12, :, :]
        
        
#         return features, target, mask

#     def __init__(self, path):
#         # super().__init__(batch_size = batch_size, shuffle = shuffle)
#         self.path = path
#         self.list_IDs = os.listdir(self.path)
#         self.labels = [i for i in range(len(self.list_IDs))]

#     def __len__(self):
#         return len(self.list_IDs)

#     def __getitem__(self, index):
#         ID = self.list_IDs[index]

#         features, target, mask = self.get_feat_label(self.load_all(self.path + ID))
#         # Load data and get label
#         # X = torch.load('data/' + ID + '.pt')
#         # y = self.labels[ID]

#         sample = {"features": features, "target": target, "mask": mask, "id": ID}
#         return sample