import argparse
import numpy as np
from data_loader import get_data_loader

from shaping import ImgGANShaping

from torchvision import utils


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
    train_loader, test_loader = get_data_loader(args)
    # feature_extraction = FeatureExtractionTest(train_loader, test_loader, args.cuda, args.batch_size)

    train_loader = get_infinite_batches(train_loader)
    for i in range(args.generator_iters):
        images = train_loader.__next__()
        dloss, gloss = model.train(
            {"o": images, "g": np.empty(
                (args.batch_size, 0)), "u": np.empty((args.batch_size, 0))}
        )
        if i % 100 == 0:
            model.evaluate()
            grid = utils.make_grid(images)
            # print("Grid of 8x8 images saved to 'dgan_model_image.png'.")
            utils.save_image(grid, "dgan_model_image_grid_V2.png")
            print('Disc loss: ', dloss)


if __name__ == "__main__":
    """
    python test_gan_img.py --model WGAN-GP \
               --is_train True \
               --download True \
               --dataroot datasets/cifar \
               --dataset cifar \
               --generator_iters 40000 \
               --cuda True \
               --batch_size 64
    """
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--generator_iters', default=40000)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--dataset', default='cifar')
    parser.add_argument('--dataroot', default='GAN_exps/datasets/cifar')
    parser.add_argument('--download', default=False, action='store_true')

    args = parser.parse_args()
    main(args)
