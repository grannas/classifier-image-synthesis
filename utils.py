import matplotlib.pyplot as plt
import torch as ch
from robustness import imagenet_models, cifar_models
import dill


def display_image(image, description=''):
    image = image / 255
    if len(image.shape) == 4:
        plt.imshow(image[0])
    else:
        plt.imshow(image)
    plt.title(description)
    plt.axis('off')
    plt.show()


def get_pretrained_model(model_name='', use_cpu=True):
    """
    Usage:

    M1, M11 = get_pretrained_model('CiFaR')
    M2, M22 = get_pretrained_model('imagenet')
    M3, M33 = get_pretrained_model('restrictedimagenet')
    M4, M44 = get_pretrained_model('h2z')
    M5, M55 = get_pretrained_model('a2o')
    M6, M66 = get_pretrained_model('s2w')
    """

    arch = 'resnet50'
    model_path = 'Models/'
    resume_path = model_path + model_name + '.pt'

    # Different model setup depending on which dataset it was trained on
    if model_name.lower() == 'cifar':
        raw_model = cifar_models.__dict__[arch](num_classes=10)

    elif model_name.lower() == 'imagenet':
        raw_model = imagenet_models.__dict__[arch](num_classes=1000)

    elif model_name.lower() == 'restrictedimagenet':
        raw_model = imagenet_models.__dict__[arch](num_classes=9)

    elif model_name.lower() == 'h2z':
        raw_model = imagenet_models.__dict__[arch](num_classes=2)

    elif model_name.lower() == 'a2o':
        raw_model = imagenet_models.__dict__[arch](num_classes=2)

    elif model_name.lower() == 's2w':
        raw_model = imagenet_models.__dict__[arch](num_classes=2)

    else:
        print("Unknown model.")
        return None, None

    # ot really implemented yet
    if use_cpu:
        checkpoint = ch.load(resume_path, pickle_module=dill, map_location=ch.device('cpu'))
    else:
        checkpoint = ch.load(resume_path, pickle_module=dill)

    # Models are stored with different naming conventions
    state_dict_path = 'model'
    if not ('model' in checkpoint):
        state_dict_path = 'state_dict'

    sd = checkpoint[state_dict_path]

    state_dict1 = {}
    state_dict2 = {}

    # Sorting out the unnecessary things in the checkpoint
    for k, v in sd.items():
        words = k.split('.')

        # Need to remove the module.attacker.model. text
        if words[2] == 'model':
            state_dict1[k[len('module.attacker.model.'):]] = v

        # Two models states are actually stored, but they seem to be identical
        # if words[1] == 'model':
        #     state_dict2[k[len('module.model.'):]] = v

    raw_model.load_state_dict(state_dict1)

    # Loaded model does not have a final activation layer for some reason
    return ch.nn.Sequential(raw_model, ch.nn.Softmax(1)), raw_model
