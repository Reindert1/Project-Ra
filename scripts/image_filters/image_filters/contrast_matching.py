from skimage import exposure
from matplotlib import pyplot as plt
import argparse
import cv2


def argparser():
    parser = argparse.ArgumentParser(description='Overlay multiple images')
    parser.add_argument("--source", "-s", help="Source filename",
                        required=True)

    parser.add_argument('--reference', "-r",
                        help="Reference image", required=True)

    parser.add_argument("--output", "-O", help="Set output path", required=True)

    args = parser.parse_args()
    return args


def process(source_path, reference_path, output_path):
    print("[INFO] Loading images")
    source = cv2.imread(source_path)
    reference = cv2.imread(reference_path)

    # check for performing multichannel histogram matching

    print("[INFO] Performing matching")
    multi = True if source.shape[-1] > 1 else False
    matched = exposure.match_histograms(source, reference, channel_axis=multi)
    cv2.imwrite(output_path, matched)

    # Create histograms
    create_histogram(source, reference, matched, "histogram.jpg")

    print("[INFO] Done")


def create_histogram(source, reference, matched, name):
    print(f"[INFO] Creating Histogram for {name}")
    color = ('b', 'g', 'r')
    plt.figure()
    for i, col in enumerate(color):
        histr = cv2.calcHist([source], [i], None, [256], [0, 256])
        plt.plot(histr, color="b", label="Source")
        histr = cv2.calcHist([reference], [i], None, [256], [0, 256])
        plt.plot(histr, color="r", label="Reference")
        histr = cv2.calcHist([matched], [i], None, [256], [0, 256])
        plt.plot(histr, color="g", label="Matched")
        plt.xlim([0, 256])
    plt.legend()
    plt.savefig(name)


if __name__ == '__main__':
    args = argparser()
    process(args.source, args.reference, args.output)
