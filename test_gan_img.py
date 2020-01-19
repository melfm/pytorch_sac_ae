import argparse
import numpy as np
import os
import utils

from os import listdir

from data_loader import get_data_loader
from shaping import ImgGANShaping

from torchvision import utils as torch_utils


def get_infinite_batches(data_loader):
    while True:
        for i, (images, _) in enumerate(data_loader):
            yield images


def main(args):

    model = ImgGANShaping(
        dims={"o": (3, 32, 32), "g": (0,), "u": (0,)},
        max_u=1.0,
        gamma=0.99,
        layer_sizes=[256, 256],
        potential_weight=1.0,
        norm_obs=True,
        norm_eps=0.01,
        norm_clip=5,
    )

    # Load datasets to train and test loaders
    train_loader, _ = get_data_loader(args)
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    img_dir = utils.make_dir(os.path.join(args.work_dir, 'images'))
    real_images = utils.make_dir(os.path.join(img_dir, 'real'))
    generated_images = utils.make_dir(os.path.join(img_dir, 'fake'))

    training_step = 0
    if args.restore:
        # Get the latest model
        all_files = listdir(model_dir)
        disc_models = [
            model for model in all_files if model.startswith('discriminator')]
        ids = [id.split('_')[1].split('.pt')[0]
               for id in disc_models if id.endswith('.pt')]
        step = max([int(id) for id in ids])
        training_step = step
        model.load(model_dir, step)
        print('Continuing from training step ', training_step)

    train_loader = get_infinite_batches(train_loader)
    for i in range(training_step, args.generator_iters):
        images = train_loader.__next__()
        dloss, gloss = model.train(
            {"o": images, "g": np.empty(
                (args.batch_size, 0)), "u": np.empty((args.batch_size, 0))}
        )
        if i % 100 == 0:
            model.evaluate(generated_images, i)
            grid = torch_utils.make_grid(images)
            img_file = real_images + \
                "/real_image_grid_" + str(i) + ".png"
            torch_utils.save_image(grid, img_file)
            print('Discriminator loss: ', dloss.item())
            print('Generator loss: ', gloss)
            # Store the model
            model.save(model_dir, i)


if __name__ == "__main__":
    """
    python test_gan_img.py --model WGAN-GP \
               --is_train True \
               --download True \
               --dataroot datasets/cifar \
               --dataset cifar \
               --generator_iters 40000 \
               --cuda True \
               --batch_size 64 \
               --work_dir gan_image_exp
    """
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--generator_iters', default=40000)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--dataset', default='cifar')
    parser.add_argument('--dataroot', default='GAN_exps/datasets/cifar')
    parser.add_argument('--download', default=False, action='store_true')
    parser.add_argument('--work_dir', type=str, default='gan_image_exp')
    parser.add_argument('--restore', default=False, action='store_true')

    args = parser.parse_args()
    main(args)
