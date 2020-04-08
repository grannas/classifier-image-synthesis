import torch as ch
from torch.autograd import Variable
import torch.nn as nn


def pgd_attack(img, eps, model, steps=5, step_size=.1, targeted_attack=False, target_class=0, norm='l2',
               verbose=True):
    # Based on the tutorial: https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
    """
    img:   should have shape (1, 3, imgsize, imgsize)
    model: the pretrained model WITHOUT softmax layer at end
    """
    img.requires_grad = True
    out = model(img)

    if targeted_attack:
        sign_direction = -1
    else:
        # Use predicted class as "anti-target" to minimize
        _, temp_target = ch.max(out, 1)
        target_class = temp_target[0]
        sign_direction = 1

    if verbose:
        print("Target_before", out[0, target_class])

    # Creates target class that the loss function accepts
    target = ch.FloatTensor(out.shape).zero_()
    target[0, target_class] = 1.
    _, targets = target.max(dim=1)

    # The attack
    pert_img = img
    for _ in range(steps):
        pert_img.requires_grad = True

        output = model(pert_img)
        loss = nn.CrossEntropyLoss()(output, Variable(targets))

        model.zero_grad()
        loss.backward()
        grad = pert_img.grad.data

        # Fast gradient sign method
        pert_img = pert_img + sign_direction * step_size * grad.sign()

        # Making sure pert image is inside l-2 or l-ing ball
        diff = pert_img - img
        if (norm == 'l2'):
            diff = ch.renorm(diff, 2, 0, eps)
        else:
            diff = ch.clamp(diff, -eps, eps)
        pert_img = img + diff


        pert_img = Variable(ch.clamp(pert_img, 0, 1))

    out = model(pert_img)

    if verbose:
        print("Output after:", ch.max(out, 1))
        print("Target after", out[0, target_class])

    return pert_img
