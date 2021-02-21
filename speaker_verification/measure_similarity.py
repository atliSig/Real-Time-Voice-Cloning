import os
import argparse
import numpy as np

def cos_sim(x, y):
    return np.dot(x, y)/(np.linalg.norm(x) * np.linalg.norm(y))

if __name__ == '__main__':
    ## Info & args
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--embed_dir",
        default='./speaker_verification/data/embeddings',
        help="A path to a directory where generated embeddings will be stored")
    parser.add_argument("--out_path",
        default='./speaker_verification/data/results.tsv',
        help="A path to a TSV file where generated results will be written")
    args = parser.parse_args()

    gt_dir = os.path.join(args.embed_dir, 'ground_truth_audio')
    synth_dir = os.path.join(args.embed_dir, 'synthesized_audio')

    assert os.path.isdir(gt_dir), 'Ground truth audio directory not found'
    assert os.path.isdir(synth_dir), 'Synthesized audio directory not found'

    out_file = open(args.out_path, 'w')

    for gt_subdir, gt_dirs, gt_files in os.walk(gt_dir):
        for gt_audio in gt_files:
            ground_truth = np.load(os.path.join(gt_subdir, gt_audio))
            gt_id = os.path.splitext(gt_audio)[0]
            gt_speaker = gt_subdir

            for s_subdir, s_dirs, s_files in os.walk(synth_dir):
                for s_audio in s_files:
                    synthesized = np.load(os.path.join(s_subdir, s_audio))
                    s_id = os.path.splitext(s_audio)[0]
                    s_speaker = s_subdir

                    similarity = cos_sim(ground_truth, synthesized)

                    out_file.write(f'{gt_id}\t{gt_speaker}\t{s_id}\t{s_speaker}\t{similarity}\n')

