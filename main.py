PATH = '/opt/project/'
SAMPLE_PATH = '/opt/project/abstract_samples'
NUM_SPLITS = 3  # number of data splits to comute std of DSD
SAMPLES_PER_SPLIT = 500  # number of samples in a single DSD run.
# We recommend at least 10k samples for evaluation to get reasonable estimates.
AUDIO_LENGTH = 2  # length of individual sample, in seconds
NUM_NOISE_LEVELS = 3  # number of different noise levels for samples to evaluate

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import audio_distance
from sample_utils import subsample_audio
import os

def main():
    # subsample_audio(os.path.join(PATH, 'abstract.wav'),
    #                 SAMPLE_PATH,
    #                 num_samples=NUM_SPLITS * SAMPLES_PER_SPLIT,
    #                 length=AUDIO_LENGTH,
    #                 num_noise_levels=NUM_NOISE_LEVELS)

    reference_path = os.path.join(SAMPLE_PATH, 'ref', '*.wav')
    eval_paths = [os.path.join(SAMPLE_PATH, f'noisy_{i+1}', '*.wav') for i
                  in range(NUM_NOISE_LEVELS)]

    evaluator = audio_distance.AudioDistance(
        load_path=os.path.join(PATH, 'checkpoint', 'ds2_large', 'model.ckpt-54800'),
        meta_path=os.path.join(PATH, 'checkpoint', 'collection-stripped-meta.meta'),
        required_sample_size=NUM_SPLITS * SAMPLES_PER_SPLIT,
        num_splits=NUM_SPLITS)

    evaluator.load_real_data(reference_path)

    dist_names = ['FDSD', 'KDSD', 'cFDSD', 'cKDSD']
    def print_results(values):
      print('\n' + ', '.join(['%s = %.5f (%.5f)' % (n, v[0], v[1]) for n, v
                              in zip(dist_names, values)]))

    with tf.Session(config=evaluator.sess_config) as sess:
      print('Computing reference DeepSpeech distances.')
      values = evaluator.get_distance(sess=sess)
      print_results(values)
      distances = [values]

      for eval_path in eval_paths:
        print('\nComputing DeepSpeech distances for files in the directory:\n'
              + os.path.dirname(eval_path))
        values = evaluator.get_distance(sess=sess, files=eval_path)
        print_results(values)
        distances.append(values)

if __name__ == '__main__':
    main()