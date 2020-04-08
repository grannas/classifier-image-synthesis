import tensorflow as tf

from utils import display_image


class ProjectedGradientDescent:
    def __init__(self, model, orig_image, target_label, iters, lr, eps, targeted=True):
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.iters = iters
        self.norm = 'l2'
        self.lr = lr
        self.eps = eps
        self.orig_image = orig_image
        self.model = model
        self.target_label = target_label
        self.targeted = targeted

    def step(self, image):
        with tf.GradientTape() as tape:
            tape.watch(image)
            pred = self.model(image)
            loss = self.loss(self.target_label, pred)

        grad = tape.gradient(loss, image)
        signed_grad = tf.sign(grad)

        if self.targeted:
            image -= self.lr * signed_grad
        else:
            image += self.lr * signed_grad

        diff = (image - self.orig_image)

        if self.norm == 'l2':
            diff = tf.clip_by_norm(diff, self.eps)
        elif self.norm == 'inf':
            diff = tf.clip_by_value(diff, -self.eps, self.eps)

        image = tf.clip_by_value(self.orig_image + diff, 0., 1.)

        return image

    def gen_adv_example(self, showImage=False):
        if showImage:
            display_image(self.orig_image, "Orig image")

        image = self.orig_image
        for i in range(self.iters):
            image = self.step(image)

        if showImage:
            display_image(image, "Orig image")

        return image
