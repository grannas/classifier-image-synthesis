from keras.datasets import cifar10
import numpy as np
import torchvision.transforms as transforms
import torch as ch
import os
from PIL import Image

from utils import display_image


class Generator:
    def __init__(self):
        self.means = []
        self.covs = []
        self.orig_shape = None

    def calc_class_stats(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        self.orig_shape = x_train[0].shape

        for k in range(10):
            bol = (y_train == k).reshape(-1)
            sub_x = x_train[bol]
            mean = np.mean(sub_x, axis=0).flatten()
            sub_x = sub_x.reshape(sub_x.shape[0], -1)
            cov = (sub_x - mean).T @ (sub_x - mean) / len(mean)

            self.means.append(mean)
            self.covs.append(cov)

    def generate_seed(self, class_id):
        sample = np.rint(np.random.multivariate_normal(self.means[class_id], self.covs[class_id])).astype(int).reshape(
            self.orig_shape)
        sample = np.clip(sample, 0, 255)
        return sample

    def visualize_example_seeds(self, count=3):
        for k in range(count):
            sample = self.generate_seed(k)
            display_image(sample, "Example seed from class {}".format(k))


class SeedGenerator():
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.mean = None
        self.sigma = None
        self.orig_shape = None

        # Cropping to size 56 to easily scale up to 224
        self.transform = transforms.Compose([
            transforms.CenterCrop(56),
            transforms.ToTensor(),
        ])

    def load_data(self):
        files = os.listdir(self.data_path)
        count = 0
        for file in files:
            count += 1
            image = Image.open(self.data_path + file)
            # image = image.convert("RGB")

            if self.data is None:
                self.data = self.transform(image).expand(1, -1, -1, -1)
                self.orig_shape = self.data.shape

                # self.data = self.data.view(1, self.data.shape[0], self.data.shape[1], self.data.shape[2])
            else:
                img_t = self.transform(image).expand(1, -1, -1, -1)
                if img_t.shape != self.orig_shape:
                    pass
                else:
                    self.data = ch.cat((self.data, img_t), 0)

    def generate_dist(self):
        self.mu = ch.mean(self.data, 0).flatten()
        data_flat = self.data.reshape(self.data.shape[0], -1)
        self.sigma = (data_flat - self.mu).T @ (data_flat - self.mu) / (len(self.mu) / 20) + 1e-3 * ch.eye(len(self.mu))

        print(self.mu)
        print(len(self.mu))

    def sample(self):
        m = ch.distributions.multivariate_normal.MultivariateNormal(self.mu, self.sigma)
        sample = m.sample().view(self.orig_shape)
        sample = ch.clamp(sample, 0., 1.)
        return ch.nn.Upsample(size=224, mode='nearest')(sample)
