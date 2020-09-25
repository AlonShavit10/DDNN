import random
import numpy as np
import tensorflow as tf

class data_and_gt():
    def __init__(self,image_code,image_regular,gt):
        self.image_code=image_code
        self.image_regular = image_regular
        self.gt=gt

class DataGenerator():
    def __init__(self, data_and_gt_array, code_min_val, code_max_val, mode):
        self.mode = mode
        self.code_min_val=code_min_val
        self.code_max_val=code_max_val
        random.seed(42)
        random.shuffle(data_and_gt_array)
        self.data_and_gt_array=data_and_gt_array



    def norm_code_range(self, value):
        if self.mode == "1":
            inx = 0
        elif self.mode == "2":
            inx = 1
        elif self.mode == "3":
            inx = 2
        elif self.mode == "4":
            inx = 3
        elif self.mode == "12":
            inx = 4
        elif self.mode == "123":
            inx = 5
        elif self.mode == "1234":
            inx = 6
        elif self.mode == "img":
            inx = 6
        else:
            raise NameError

        dynamic_range_Code = self.code_max_val[0][inx] - self.code_min_val[0][inx]
        return np.float32(np.float32(value - self.code_min_val[0][inx]) / dynamic_range_Code)


    def get_img_according_to_mode(self, cur_data:data_and_gt):
        # TODO delete
        # mean_val = np.mean(cur_data.image_code[3])
        # print(mean_val)
        if self.mode == "1":
            img = self.norm_code_range(cur_data.image_code[0])
        elif self.mode == "2":
            img = self.norm_code_range(cur_data.image_code[1])
        elif self.mode == "3":
            img = self.norm_code_range(cur_data.image_code[2])
        elif self.mode == "4":
            img = self.norm_code_range(cur_data.image_code[3])
        elif self.mode == "12":
            img = self.norm_code_range(cur_data.image_code[4])
        elif self.mode == "123":
            img = self.norm_code_range(cur_data.image_code[5])
        elif self.mode == "1234":
            img = self.norm_code_range(cur_data.image_code[6])
        elif self.mode == "img":
            img = np.float32(cur_data.image_regular)/255
        else:
            raise NameError

        # TODO delete
        # mean_val = np.mean(img)
        # print(mean_val)
        return img

    @staticmethod
    def to_tensor(input):
        output = tf.convert_to_tensor(input, dtype=tf.float32)
        output = tf.expand_dims(output, -1)  # TODO check
        return output

    def generate_images(self, batch_size=16, inf_looping=True):
        images, gts = [], []


        while True:
            for idx in range(0,self.data_and_gt_array.__len__()):
                cur_data= self.data_and_gt_array[idx]
                img = self.get_img_according_to_mode(cur_data)
                gt = cur_data.gt
                images.append(img)
                gts.append(gt)


                # yielding condition
                if len(images) >= batch_size:
                    images_tensor = np.array(images)
                    images_tensor = self.to_tensor(images_tensor)
                    gt_out = (np.array(gts).astype(np.float32))

                    yield images_tensor,  gt_out
                    images, gts = [], []

            if not inf_looping:
                break
