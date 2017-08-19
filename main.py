import argparse
import os
import glob
from tqdm import tqdm

from configs import MUSIC_DIR, QUERY_PATH, FEATURES_DIR
from ops import query2wav, music2csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Feature Extractor")

    parser.add_argument('--download', action='store_true', default=False,
        help="""Download wav files from YouTube/SoundCloud search results.
        Queries are stored in ./{}.
        Wav files are stored in ./{}.""".format(os.path.basename(QUERY_PATH), os.path.basename(MUSIC_DIR)))
    parser.add_argument('--feature', action='store_true', default=False,
        help="""Extract features from music files in ./{}.
        Features are stored in ./{}""".format(os.path.basename(MUSIC_DIR), os.path.basename(FEATURES_DIR)))

    args = parser.parse_args()

    # Downalod wav files.
    if args.download:
        query2wav('./query.txt')

    # Extract music features.
    if args.feature:

        music_filenames = glob.glob(os.path.join(MUSIC_DIR, '*.wav'))

        for music_filename in tqdm(music_filenames):
            music_name = os.path.basename(music_filename)[:-4] # Remove '.wav'
            music2csv(music_path = music_filename,
                      csv_path = os.path.join(FEATURES_DIR, music_name+'.csv'),
                      start = 0.0,
                      end = 3.0,
                      tss = 0.1,
                      npca_comp = 30)
