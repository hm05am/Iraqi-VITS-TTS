"""
Data loading utilities for VITS Iraqi Arabic TTS.

Reads filelist format: wav_path|text
Handles UTF-8 Arabic text correctly.
"""

import os
import random
import torch
import torch.utils.data
import numpy as np
import librosa

from text import text_to_sequence
from mel_processing import spectrogram
from utils import load_filepaths_and_text
import commons


class TextAudioLoader(torch.utils.data.Dataset):
    """
    Dataset that loads (text, spectrogram, audio) triplets.
    
    Filelist format (UTF-8):
        path/to/audio.wav|Arabic text goes here
    
    Args:
        filelist_path: Path to filelist.
        hparams: HParams object with data configuration.
    """
    def __init__(self, filelist_path, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(filelist_path)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.add_blank = getattr(hparams, 'add_blank', True)

        # Filter out entries with missing audio
        self._filter()
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)

    def _filter(self):
        """Filter out entries where audio files don't exist."""
        filtered = []
        for entry in self.audiopaths_and_text:
            audio_path = entry[0]
            if os.path.isfile(audio_path):
                filtered.append(entry)
            else:
                print(f"Warning: Audio file not found, skipping: {audio_path}")
        self.audiopaths_and_text = filtered
        print(f"Dataset: {len(self.audiopaths_and_text)} valid entries")

    def get_audio_text_pair(self, audiopath_and_text):
        """Load and process one (text, spec, wav) pair."""
        audiopath = audiopath_and_text[0]
        text = audiopath_and_text[1]

        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        return (text, spec, wav)

    def get_audio(self, filename):
        """Load audio file and compute spectrogram."""
        # Load with librosa (handles resampling)
        audio, sr = librosa.load(filename, sr=self.sampling_rate)
        audio = audio / max(abs(audio.max()), abs(audio.min())) * 0.95  # Normalize
        audio = torch.FloatTensor(audio).unsqueeze(0)

        spec = spectrogram(audio, self.filter_length, self.hop_length,
                           self.win_length, center=False)
        spec = torch.squeeze(spec, 0)
        return spec, audio.squeeze(0)

    def get_text(self, text):
        """Convert text string to integer sequence."""
        text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextAudioCollate:
    """
    Collate function that pads variable-length sequences in a batch.
    Returns padded tensors and their lengths.
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        # Sort by spectrogram length (descending) for efficient packing
        batch = sorted(batch, key=lambda x: x[1].size(1), reverse=True)

        # Get max lengths
        max_text_len = max(x[0].size(0) for x in batch)
        max_spec_len = max(x[1].size(1) for x in batch)
        max_wav_len = max(x[2].size(0) for x in batch)

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len).zero_()
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len).zero_()
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len).zero_()

        for i, (text, spec, wav) in enumerate(batch):
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav_padded[i, 0, :wav.size(0)] = wav

        if self.return_ids:
            ids = torch.LongTensor([i for i in range(len(batch))])
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, ids

        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Bucket sampler that groups similar-length utterances together.
    Minimizes padding waste within each batch.
    
    For single-GPU, use num_replicas=1 and rank=0.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=1, rank=0,
                 shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = [dataset.get_audio_text_pair(dataset.audiopaths_and_text[i])[1].size(1)
                        for i in range(len(dataset))]
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        # Remove empty buckets
        buckets = [b for b in buckets if len(b) > 0]

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices_bucket = list(bucket)
                random.Random(self.epoch).shuffle(indices_bucket)
                # Pad to fill last batch
                while len(indices_bucket) % (self.num_replicas * self.batch_size) != 0:
                    indices_bucket.append(random.choice(bucket))
                indices.extend(indices_bucket)
        else:
            for bucket in self.buckets:
                indices.extend(list(bucket))

        # Subsample for this rank
        indices = indices[self.rank::self.num_replicas]
        return iter(indices)

    def _bisect(self, x):
        lo, hi = 0, len(self.boundaries) - 1
        if x < self.boundaries[0]:
            return 0
        if x >= self.boundaries[-1]:
            return len(self.boundaries) - 2
        while hi > lo + 1:
            mid = (hi + lo) // 2
            if x < self.boundaries[mid]:
                hi = mid
            else:
                lo = mid
        return lo

    def __len__(self):
        return self.num_samples
