import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import data.util as util
from pytorch_wavelets import DWTForward, DWTInverse
from PIL import Image
import json
from scipy.ndimage import convolve, gaussian_filter


class HiFaceGAN_Dataset(data.Dataset):
    '''
    Read LR and HR image pairs.
    If only HR image is provided, generate LR image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(HiFaceGAN_Dataset, self).__init__()
        self.opt = opt
        self.paths_fake_LR = None
        self.paths_HR = None
        self.paths_real_LR = None
        self.LR_env = None  # environment for lmdb
        self.HR_env = None

        # read image list from subset list txt
        if opt['subset_file'] is not None and opt['phase'] == 'train':
            with open(opt['subset_file']) as f:
                self.paths_HR = sorted([os.path.join(opt['dataroot_HR'], line.rstrip('\n'))
                                        for line in f])
            if opt['dataroot_LR'] is not None:
                raise NotImplementedError(
                    'Now subset only supports generating LR on-the-fly.')
        else:  # read image list from lmdb or image files
            self.HR_env, self.paths_HR = util.get_image_paths(
                opt['data_type'], opt['dataroot_HR'])
            self.LR_env, self.paths_fake_LR = util.get_image_paths(
                opt['data_type'], opt['dataroot_fake_LR'])
            self.LR_env, self.paths_real_LR = util.get_image_paths(
                opt['data_type'], opt['dataroot_real_LR'])
            self.LR_env, self.paths_fake_weights = util.get_image_paths(
                opt['data_type'], opt['dataroot_fake_weights'])  # Domain Distance Map
            self.LR_env, self.paths_real_weights = util.get_image_paths(
                opt['data_type'], opt['dataroot_real_weights'])

        assert self.paths_HR, 'Error: HR path is empty.'
        # if self.paths_LR and self.paths_HR:
        #     assert len(self.paths_LR) == len(self.paths_HR), \
        #         'HR and LR datasets have different number of images - {}, {}.'.format(\
        #         len(self.paths_LR), len(self.paths_HR))

        self.random_scale_list = [1]
        self.DWT2 = DWTForward(J=1, wave='haar', mode='symmetric')
        path_json_lm = os.path.join(self.opt['path_landmarks'])
        with open(path_json_lm) as f:
            self.landmarks_json = json.load(f)

    def __getitem__(self, index):
        # HR_path, LR_path = None, None fake real
        scale = self.opt['scale']
        HR_size = self.opt['HR_size']
        index_fake, index_real = index, np.random.randint(
            0, len(self.paths_real_LR))
        # get LR image
        LR_fake_path = self.paths_fake_LR[index_fake]
        LR_real_path = self.paths_real_LR[index_real]
        fake_w_path = self.paths_fake_weights[index_fake]

        img_LR_fake = util.read_img(self.LR_env, LR_fake_path)
        img_LR_real = util.read_img(self.LR_env, LR_real_path)
        fake_w = np.load(fake_w_path)[0].transpose((1, 2, 0))

        fake_w = cv2.resize(
            fake_w, (img_LR_fake.shape[1], img_LR_fake.shape[0]), interpolation=cv2.INTER_LINEAR)

        fake_w = np.reshape(fake_w, list(fake_w.shape)+[1])
        # get HR image
        HR_path = self.paths_HR[index_fake]

        index_unpair = np.random.randint(0, len(self.paths_HR))
        HR_unpair = self.paths_HR[index_unpair]

        img_HR = util.read_img(self.HR_env, HR_path)
        img_unpair_HR = util.read_img(self.HR_env, HR_unpair)

        face_mask = None

        # from PIL import Image
        # import numpy as np
        # c = Image.fromarray(np.uint8(img_LR_fake*255))
        # d = Image.fromarray(np.uint8(img_LR_real*255))
        # c.show()
        # d.show()
        # print()
        # modcrop in the validation / test phase
        if self.opt['phase'] != 'train':
            img_HR = util.modcrop(img_HR, scale)
        # change color space if necessary
        if self.opt['color']:
            img_HR = util.channel_convert(
                img_HR.shape[2], self.opt['color'], [img_HR])[0]

        if self.opt['phase'] == 'train':
            # Create landmarks
            lm = self.landmarks_json[str(index)]['image']['face_landmarks']
            lm = np.asarray(lm).astype(int)
            base_cnv = np.zeros((1024, 1024)).astype(np.uint8)

            # Base Contour
            #ctr = lm[:17].reshape((-1,1,2)).astype(np.int32)
            lm_out = np.concatenate((lm[:17], np.flip(lm[22:27], axis=0), np.array(
                [lm[27]]), np.flip(lm[17:22], axis=0)), axis=0)
            ctr = lm_out.reshape((-1, 1, 2)).astype(np.int32)
            cv2.drawContours(base_cnv, [ctr], -1,
                             (255, 255, 255), thickness=cv2.FILLED)
            base_cnv = base_cnv.astype(int)/4
            base_cnv = cv2.resize(base_cnv, dsize=(512, 512),
                                  interpolation=cv2.INTER_CUBIC)

            face_mask = np.zeros((1024, 1024))
            kernel = np.ones((20, 20))
            for lpt in lm[28:]:
                face_mask[lpt[1], lpt[0]] = 255
            face_mask = convolve(face_mask, kernel)
            face_mask = gaussian_filter(face_mask, sigma=30)
            face_mask = cv2.resize(face_mask, (512, 512))*255/np.max(face_mask)
            face_mask = base_cnv+face_mask
            face_mask = gaussian_filter(face_mask, sigma=10)*1.2
            face_mask = np.clip(face_mask, 40, 255)/255
            # if the image size is too small
            H, W, _ = img_HR.shape
            if H < HR_size or W < HR_size:
                img_HR = cv2.resize(
                    np.copy(img_HR), (HR_size, HR_size), interpolation=cv2.INTER_LINEAR)
                # using matlab imresize
                img_LR = util.imresize_np(img_HR, 1 / scale, True)
                if img_LR.ndim == 2:
                    img_LR = np.expand_dims(img_LR, axis=2)

            H, W, C = img_LR_fake.shape
            H_r, W_r, C = img_LR_real.shape

            LR_size = HR_size // scale

            # randomly crop [Applicable in DASR due to very HR image, on face sr seems not important]
            rnd_h_fake = random.randint(0, max(0, H - LR_size))
            rnd_w_fake = random.randint(0, max(0, W - LR_size))
            rnd_h_real = random.randint(0, max(0, H_r - LR_size))
            rnd_w_real = random.randint(0, max(0, W_r - LR_size))
            img_LR_fake = img_LR_fake[rnd_h_fake:rnd_h_fake +
                                      LR_size, rnd_w_fake:rnd_w_fake + LR_size, :]
            img_LR_real = img_LR_real[rnd_h_real:rnd_h_real +
                                      LR_size, rnd_w_real:rnd_w_real + LR_size, :]
            fake_w = fake_w[rnd_h_fake:rnd_h_fake +
                            LR_size, rnd_w_fake:rnd_w_fake + LR_size, :]

            H, W, C = img_HR.shape
            H_real, W_real, C_real = img_unpair_HR.shape

            rnd_h = int(rnd_h_fake*scale)
            rnd_w = int(rnd_w_fake*scale)
            rnd_h_real = random.randint(0, max(0, H_real - HR_size))
            rnd_w_real = random.randint(0, max(0, W_real - HR_size))
            img_HR = img_HR[rnd_h:rnd_h + HR_size, rnd_w:rnd_w + HR_size, :]
            img_unpair_HR = img_unpair_HR[rnd_h_real:rnd_h_real +
                                          HR_size, rnd_w_real:rnd_w_real + HR_size, :]
            face_mask = np.expand_dims(
                face_mask[rnd_h:rnd_h + HR_size, rnd_w:rnd_w + HR_size], axis=2)

            # augmentation - flip, rotate => augment the image here
            img_LR_fake, img_LR_real, img_HR, img_unpair_HR, fake_w, face_mask \
                = util.augment([img_LR_fake, img_LR_real, img_HR, img_unpair_HR, fake_w, face_mask],
                               self.opt['use_flip'], self.opt['use_rot'],self.opt['use_aug'])

            # if self.paths_real_weights:
            #     real_w = util.augment([real_w],
            #                           self.opt['use_flip'], self.opt['use_rot'])

        # change color space if necessary
        if self.opt['color']:
            img_LR = util.channel_convert(C, self.opt['color'], [img_LR])[
                0]  # TODO during val no definetion

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_HR.shape[2] == 3:
            img_HR = img_HR[:, :, [2, 1, 0]]
            img_unpair_HR = img_unpair_HR[:, :, [2, 1, 0]]
            img_LR_real = img_LR_real[:, :, [2, 1, 0]]
            img_LR_fake = img_LR_fake[:, :, [2, 1, 0]]

        img_HR_temp = np.ascontiguousarray(np.transpose(img_HR, (2, 0, 1)))
        img_unpair_HR_temp = np.ascontiguousarray(
            np.transpose(img_unpair_HR, (2, 0, 1)))
        img_LR_real_temp = np.ascontiguousarray(
            np.transpose(img_LR_real, (2, 0, 1)))
        img_LR_fake_temp = np.ascontiguousarray(
            np.transpose(img_LR_fake, (2, 0, 1)))

        img_HR = np.zeros((np.shape(img_HR_temp)[0], HR_size, HR_size))
        img_HR[:, :np.shape(img_HR_temp)[1], :np.shape(
            img_HR_temp)[2]] = img_HR_temp
        img_unpair_HR = np.zeros(
            (np.shape(img_unpair_HR_temp)[0], HR_size, HR_size))
        img_unpair_HR[:, :np.shape(img_unpair_HR_temp)[1], :np.shape(
            img_unpair_HR_temp)[2]] = img_unpair_HR_temp

        img_HR = torch.from_numpy(img_HR).float()
        img_unpair_HR = torch.from_numpy(img_unpair_HR).float()
        img_LR_real = torch.from_numpy(img_LR_real_temp).float()
        img_LR_fake = torch.from_numpy(img_LR_fake_temp).float()
        fake_weights = torch.from_numpy(np.ascontiguousarray(
            np.transpose(fake_w, (2, 0, 1)))).float()
        face_mask = torch.from_numpy(np.ascontiguousarray(
            np.transpose(face_mask, (2, 0, 1)))).float()

        #img_HR = np.zeroes((np.shape(img_HR_temp)[0],HR_size,HR_size))
        # img_HR[:,:HR_size,:HR_size]=img_HR_temp
        #img_HR = np.zeroes((np.shape(img_HR_temp)[0],HR_size,HR_size))
        # img_HR[:,:HR_size,:HR_size]=img_HR_temp

        # Wavelet Trans
        # if self.opt['phase'] == 'train':
        #     img_HR = img_HR.reshape([1, img_HR.shape[0], img_HR.shape[1], img_HR.shape[2]])
        #     LL, H_cat = self.DWT2(img_HR)
        #     LL = LL[0]*0.5
        #
        #     H_cat = H_cat[0][0]*0.5+0.5
        #     w_c, C, H, W = H_cat.shape
        #     H_cat = H_cat.reshape([w_c*C, H, W])
        #     img_HR = img_HR[0].detach()

        return {'LR_real': img_LR_real, 'LR_fake': img_LR_fake, 'HR': img_HR, 'HR_unpair': img_unpair_HR,
                'LR_real_path': LR_real_path, 'LR_fake_path': LR_fake_path, 'HR_path': HR_path,
                # 'wt_LL': LL.detach(), 'Ht_cat': H_cat.detach(),
                'fake_w': fake_weights, 'face_mask': face_mask}

    def __len__(self):
        return len(self.paths_fake_LR)
