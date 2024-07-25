from utils import image_pred, plt_show
import argparse


def argparsee():
    parser = argparse.ArgumentParser()

    parser.add_argument('--modelPath',
                        type = str,
                        required=True,
                        help = 'model giriniz'
                        )

    parser.add_argument('--imagePath',
                        type = str,
                        required=True,
                        help = 'path giriniz'
                        )

    return parser.parse_args()


if __name__ == "__main__":
    args = argparsee()
    modelPath, imagePath = args.modelPath, args.imagePath
    pred, preds = image_pred(modelPath, imagePath)
    plt_show(imagePath, pred ,preds)   