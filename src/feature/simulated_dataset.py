import torch
import numpy as np
import sacred
import feature.util as util
import analysis.motif_util as motif_util
import json
import scipy.signal

dataset_ex = sacred.Experiment("sim_dataset")

@dataset_ex.config
def config():
    # Size of batches
    batch_size = 128

    # Number of batches per epoch
    num_batches = 100

    # For each input sequence in the raw data, center it and pad to this length 
    input_length = 500

    # Alphabet used in sequences
    seq_alphabet = "ACGT"

    # One-hot encoding has this depth
    input_depth = len(seq_alphabet)

    # Probability of each sequence token for background
    bg_seq_freqs = np.array([0.25, 0.25, 0.25, 0.25])

    # Maximum random distance of motif-configuration center from sequence center
    motif_center_dist_bound = 50

    # Whether or not to perform reverse complement augmentation
    revcomp = True

    # Sample this many negatives randomly for every positive example
    negative_ratio = 1

    # Likelihood of rejecting a background sequence if it has a motif match
    background_match_reject_prob = 1

    # Match score threshold for considering background sequences to have a motif
    background_match_score_thresh = 0.9

    # Number of workers for the data loader
    num_workers = 10
    
    # Seed for generating examples
    data_seed = None


class SeqSimulator:
    def __init__(
        self, motif_config_path, input_length, seq_alphabet, bg_seq_freqs,
        motif_center_dist_bound, data_seed
    ):
        """
        Samples simulated sequences given a motif configuration file.
        Arguments:
            `motif_config_path`: path to JSON file containing motif
                configurations to sample from
            `input_length`: length of input sequences to generate
            `seq_alphabet`: string containing alphabet of sequences; defaults to
                "ACGT"
            `bg_seq_freqs`: NumPy array of background frequency for each
                sequence token
            `motif_center_dist_bound`: maximum random distance of motif-
                configuration center from sequence center
            `data_seed`: seed for sampling sequences
        """

        assert len(seq_alphabet) == len(bg_seq_freqs)
        self.input_length = input_length
        self.seq_alphabet = seq_alphabet
        self.seq_alphabet_arr = np.array(list(seq_alphabet))
        self.bg_seq_freqs = bg_seq_freqs
        self.bg_seq_freqs_cumsum = \
            np.cumsum(bg_seq_freqs / np.sum(bg_seq_freqs))
        self.motif_center_dist_bound = motif_center_dist_bound

        # Import motif configs
        with open(motif_config_path, "r") as f:
            motif_configs = json.load(f)
        self.motif_dict = motif_util.import_meme_motifs(
            motif_configs["motif_files"]
        )
        self.configs = motif_configs["configs"]
        
        self.rng = np.random.default_rng(data_seed)

    def set_seed(self, data_seed):
        """
        Sets seed of RNG.
        Arguments:
            `data_seed`: integer-valued seed
        """
        self.rng = np.random.default_rng(data_seed)
    
    def _get_possible_motifs(self, configs=None):
        """
        Computes the set of motif keys which could be called for by the motif
        configuration with non-zero probability.
        Arguments:
            `configs`: a list of configurations; default is to use
                `self.configs`
        `configs` must be one of the following 3 forms:
        1) A list of strings and/or spacings specifying the motif IDs and spaces
            between them (e.g. ["GATA1", 8, "TAL1"])
        2) A list of lists, where each inner list is a set of configurations to
            be chosen from uniformly at random
        3) A list of dictionaries, where each dictionary specifies a probability
            and a configuration; dictionaries are selected according to their
            (normalized) probabilities
        Returns a set of motif keys.
        """
        # Motif configurations are defined recursively
        if configs is None:
            configs = self.configs
        
        if all(type(c) in (str, int) for c in configs) or configs == []:
            # This is a single config containing a motif specification
            # Return the set of all motif keys in it
            return set([c for c in configs if type(c) is str])
        elif all(type(c) is list for c in configs):
            # Everything is a list, so take union
            return set().union(*[
                self._get_possible_motifs(c) for c in configs
            ])
        elif all(type(c) is dict for c in configs):
            # Everything is a dictionary, so take union if the probability is
            # non-zero
            return set().union(*[
                self._get_possible_motifs(c["configs"]) for c in configs
                if c["p"] > 0
            ])
        else:
            raise ValueError("Unknown configuration type")

    def sample_random_seq(
        self, length=None, motifs_blacklist=None, match_reject_prob=0,
        match_score_thresh=0.9, match_revcomp=True
    ):
        """
        Samples a random sequence as defined by `self.seq_alphabet` and
        `self.bg_seq_freqs`.
        Arguments:
            `length`: length of sequence to sample; defaults to
                `self.input_length`
            `motif_blacklist`: an iterable of NumPy arrays of motifs to avoid
                in the sampled sequence
            `match_reject_prob`: if the sampled sequence contains any motif in
                `motif_blacklist`, then it is rejected with this probability;
                this probability may be 0
            `match_score_thresh`: match-score threshold for considering a motif
                to be present in the sequence.
            `match_revcomp`: if True, also check motif reverse complements for
                matches
        Returns a random string sequence.
        """
        if length is None:
            length = self.input_length
        
        if self.rng.random() > match_reject_prob:
            # Just sample a sequence and return it
            inds = np.searchsorted(
                self.bg_seq_freqs_cumsum, self.rng.random(length)
            )
            return "".join(self.seq_alphabet_arr[inds])
        else:
            motifs_to_match = list(motifs_blacklist)
            if match_revcomp:
                motifs_to_match.extend([np.flip(m) for m in motifs_to_match])

            # Keep searching until we find a sequence we like
            while True:
                # Sample a long sequence (10 * length arbitrarily)
                inds = np.searchsorted(
                    self.bg_seq_freqs_cumsum, self.rng.random(length * 10)
                )
                long_seq = util.seqs_to_one_hot(
                    ["".join(self.seq_alphabet_arr[inds])]
                )[0]  # Shape: L x 4
               
                # Scan for matches to get an array of scores for each motif
                all_match_scores = [
                    scipy.signal.correlate(
                        long_seq, m, mode="valid"
                    )[:, 0] / len(m)
                    for m in motifs_to_match
                ]
                # Convert to boolean arrays
                all_match_bools = [
                    scores >= match_score_thresh for scores in all_match_scores
                ]
                # Left-justify and cut off; we won't care about different
                # lengths arising from different motif sizes
                min_length = min(len(arr) for arr in all_match_bools)
                all_match_bools = [arr[:min_length] for arr in all_match_bools]
                # Take logical or
                match_bools = np.any(np.stack(all_match_bools), axis=0)
                # Get indices of matches and gaps
                match_inds = np.where(match_bools)[0]
                # Tack on index of -1 and length
                match_inds = np.pad(
                    match_inds, (1, 1), "constant",
                    constant_values=(-1, len(match_bools))
                )
                match_gaps = np.diff(match_inds)
                if not np.any(match_gaps >= length + 1):
                    # No stretch between matches is long enough; try again
                    continue
                start = match_inds[np.where(match_gaps >= length)[0][0]] + 1

                return util.one_hot_to_seqs(
                    long_seq[start : start + length][None]
                )[0]
                
    def _sample_motif_seq(self, motif):
        """
        Samples a random sequence from a motif probability matrix. The alphabet
        is assumed to be `self.seq_alphabet`.
        Arguments:
            `motif`: an L x D array of probabilities
        Returns a string of length L.
        """
        # Renormalize to ensure it sums to 1
        motif = motif / np.sum(motif, axis=1, keepdims=True)
        prob_cumsums = np.cumsum(motif, axis=1)
        return "".join([
            self.seq_alphabet_arr[np.searchsorted(probs, self.rng.random())]
            for probs in prob_cumsums
        ])

    def sample_config_seq(
        self, configs=None, motifs_blacklist=None, match_reject_prob=0,
        match_score_thresh=0.9, match_revcomp=True, return_config=False
    ):
        """
        Samples a single sequence as defined by the motif configuration.
        Arguments:
            `configs`: a list of configurations; default is to use
                `self.configs`
            `motif_blacklist`: an iterable of NumPy arrays of motifs to avoid
                in the sampled background
            `match_reject_prob`: if the sampled sequence contains any motif in
                `motif_blacklist`, then it is rejected with this probability;
                this probability may be 0
            `match_score_thresh`: match-score threshold for considering a motif
                to be present in the sequence.
            `match_revcomp`: if True, also check motif reverse complements for
                matches
            `return_config`: if True, also return the specific configuration
                which was sampled (a pair of the start position and a list which
                may contain strings and/or integers)
        `configs` must be one of the following 3 forms:
        1) A list of strings and/or spacings specifying the motif IDs and spaces
            between them (e.g. ["GATA1", 8, "TAL1"])
        2) A list of lists, where each inner list is a set of configurations to
            be chosen from uniformly at random
        3) A list of dictionaries, where each dictionary specifies a probability
            and a configuration; dictionaries are selected according to their
            (normalized) probabilities
        Returns a string of length `self.input_length`, and perhaps a pair of
        the exact configuration chosen.
        """
        # Motif configurations are defined recursively
        if configs is None:
            configs = self.configs
        
        if all(type(c) in (str, int) for c in configs) or configs == []:
            # This is a single config containing a motif specification; this
            # also includes if the config is an empty list, in which case there
            # is no motif to insert
            motif_string = ""
            for c in configs:
                if type(c) is str:
                    motif_string += self._sample_motif_seq(self.motif_dict[c])
                else:
                    motif_string += self.sample_random_seq(
                        c, motifs_blacklist, match_reject_prob,
                        match_score_thresh, match_revcomp
                    )

            # Pick random offset from center
            offset = self.rng.integers(
                -self.motif_center_dist_bound, self.motif_center_dist_bound + 1
            )

            left_pad = (self.input_length // 2) - (len(motif_string) // 2) + \
                offset
            right_pad = self.input_length - left_pad - len(motif_string)

            left_seq = self.sample_random_seq(
                left_pad, motifs_blacklist, match_reject_prob,
                match_score_thresh, match_revcomp
            )
            right_seq = self.sample_random_seq(
                right_pad, motifs_blacklist, match_reject_prob,
                match_score_thresh, match_revcomp
            )
            result = left_seq + motif_string + right_seq

            if return_config:
                return result, (left_pad, configs)
            else:
                return result

        elif all(type(c) is list for c in configs):
            # Everything is a list, so pick uniformly
            index = self.rng.integers(len(configs))
            return self.sample_config_seq(
                configs[index], motifs_blacklist, match_reject_prob,
                match_score_thresh, match_revcomp, return_config
            )

        elif all(type(c) is dict for c in configs):
            # Everything is a dictionary, so pick according to the probabilities
            probs = np.array([c["p"] for c in configs])
            probs = probs / np.sum(probs)
            index = np.searchsorted(np.cumsum(probs), self.rng.random())
            return self.sample_config_seq(
                configs[index]["configs"], motifs_blacklist, match_reject_prob,
                match_score_thresh, match_revcomp, return_config
            )
        
        else:
            raise ValueError("Unknown configuration type")

    def seqs_to_one_hot(self, seqs):
        """
        Converts a list of strings to one-hot encodings, where the encoding is
        specified by `self.seq_alphabet`.
        Arguments:
            `seqs`: a list of N strings, where every string is the same length L
        Returns an N x L x D NumPy array of one-hot encodings, in the same order
        as the input sequences.
        """
        return util.seqs_to_one_hot(
            seqs, alphabet=self.seq_alphabet, to_upper=False
        )


class SimulatedSeqDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, pos_seq_simulator, batch_size, num_batches, negative_ratio,
        neg_seq_simulator=None, revcomp=False, background_match_reject_prob=0,
        background_match_score_thresh=0.9, return_configs=False
    ):
        """
        Generates batches of one-hot-encoded sequences and binary labels.
        Arguments:
            `pos_seq_simulator (SeqSimulator): generates simulated sequences
                with motif configurations for the positive label
            `batch_size`: number of sequences per batch, B
            `num_batches`: number of batches in an epoch
            `negative_ratio`: generate this many negative sequences per batch as
                positive ones
            `neg_seq_simulator (SeqSimulator): generates simulated sequences
                with motif configurations for the negative label; by default,
                this is None, and negative sequences are randomized backgrounds
            `revcomp`: whether or not to perform revcomp to the batch; this will
                not change the batch size, but halve the number of unique
                objects per batch; if True, `batch_size` must be even
            `background_match_reject_prob`: probability of rejecting a random
                background sequence if it has a motif match; this probability
                may be 0
            `background_match_score_thresh`: match-score threshold for
                considering a background sequence to have a motif
            `return_configs`: if True, also return the specific start positions
                and configurations used to generate the sequences
        In each batch, generates a B x L x D NumPy array of one-hot encodings
        and a B-array of binary labels. May also return a B-list of
        configurations for each sequence (each is a triplet of a start position
        index, a list containing motif keys and spacings, and whether or not the
        sequence/configuration is reverse complemented).
        """
        self.pos_seq_simulator = pos_seq_simulator
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.negative_ratio = negative_ratio
        self.neg_seq_simulator = neg_seq_simulator
        self.revcomp = revcomp
        self.background_match_reject_prob = background_match_reject_prob
        self.background_match_score_thresh = background_match_score_thresh
        self.return_configs = return_configs

        if background_match_reject_prob > 0:
            self.possible_motifs = {
                key : pos_seq_simulator.motif_dict[key] for key in
                pos_seq_simulator._get_possible_motifs()
            }
            if neg_seq_simulator is not None:
                self.possible_motifs.update({
                    key : neg_seq_simulator.motif_dict[key] for key in
                    neg_seq_simulator._get_possible_motifs()
                })
        else:
            self.possible_motifs = {}

        if revcomp:
            assert batch_size % 2 == 0
            revcomp_factor = 2
        else:
            revcomp_factor = 1

        self.num_pos_per_batch = int(
            np.ceil((batch_size // revcomp_factor) / (1 + negative_ratio))
        )
        self.num_neg_per_batch = (batch_size // revcomp_factor) - \
            self.num_pos_per_batch

    def get_batch(self, index):
        """
        Returns a batch, which consists of a B x L x D NumPy array of 1-hot
        encoded sequences and a B-array of labels.
        Arguments:
            `index`: unused argument which normally would specify the index of a
                batch
        """
        labels = np.concatenate([
            np.ones(self.num_pos_per_batch), np.zeros(self.num_neg_per_batch)
        ])
      
        # Sample positive sequences
        pos_samples = [
            self.pos_seq_simulator.sample_config_seq(
                motifs_blacklist=self.possible_motifs.values(),
                match_reject_prob=self.background_match_reject_prob,
                match_score_thresh=self.background_match_score_thresh,
                match_revcomp=self.revcomp,
                return_config=self.return_configs
            ) for _ in range(self.num_pos_per_batch)
        ]
        if self.return_configs:
            pos_seqs, pos_configs = zip(*pos_samples)
            pos_seqs, pos_configs = list(pos_seqs), list(pos_configs)
        else:
            pos_seqs = pos_samples
        pos_one_hots = self.pos_seq_simulator.seqs_to_one_hot(pos_seqs)
        
        # Sample negative sequences
        if not self.num_neg_per_batch:
            # No negatives in the batch
            neg_one_hots = np.empty((0,) + pos_one_hots.shape[1:])
            if self.return_configs:
                neg_configs = []
        else:
            if self.neg_seq_simulator is None:
                neg_seqs = [
                    self.pos_seq_simulator.sample_random_seq(
                        motifs_blacklist=self.possible_motifs.values(),
                        match_reject_prob=self.background_match_reject_prob,
                        match_score_thresh=self.background_match_score_thresh,
                        match_revcomp=self.revcomp
                    ) for _ in range(self.num_neg_per_batch)
                ]
                if self.return_configs:
                    # Configurations are just empty here
                    neg_configs = [None] * len(neg_seqs)
            else:
                neg_samples = [
                    self.neg_seq_simulator.sample_config_seq(
                        motifs_blacklist=self.possible_motifs.values(),
                        match_reject_prob=self.background_match_reject_prob,
                        match_score_thresh=self.background_match_score_thresh,
                        match_revcomp=self.revcomp,
                        return_config=self.return_configs
                    ) for _ in range(self.num_neg_per_batch)
                ]
                if self.return_configs:
                    neg_seqs, neg_configs = zip(*neg_samples)
                    neg_seqs, neg_configs = list(neg_seqs), list(neg_configs)
                else:
                    neg_seqs = neg_samples
            # Convert to one-hot sequences, just use the positive simulator
            neg_one_hots = self.pos_seq_simulator.seqs_to_one_hot(neg_seqs)

        one_hots = np.concatenate([pos_one_hots, neg_one_hots])
        if self.return_configs:
            configs_no_orient = pos_configs + neg_configs

        if self.revcomp:
            # Only support reverse-complement augmentation for ACGT sequences
            assert self.pos_seq_simulator.seq_alphabet == "ACGT"
            one_hots = np.concatenate([
                one_hots, np.flip(one_hots, axis=(1, 2))
            ])
            labels = np.concatenate([labels, labels])
            if self.return_configs:
                configs = [tup + (False,) for tup in configs_no_orient] + \
                    [tup + (True,) for tup in configs_no_orient]
        else:
            if self.return_configs:
                configs = [tup + (True,) for tup in configs_no_orient]
        
        if self.return_configs:
            return one_hots, labels, configs
        else:
            return one_hots, labels

    def __iter__(self):
        """
        Returns an iterator over the batches. If the dataset iterator is called
        from multiple workers, each worker will be give a shard of the full
        range.
        """
        worker_info = torch.utils.data.get_worker_info()
        num_batches = self.num_batches
        if worker_info is None:
            # In single-processing mode
            start, end = 0, num_batches
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            shard_size = int(np.ceil(num_batches / num_workers))
            start = shard_size * worker_id
            end = min(start + shard_size, num_batches)
        return (self.get_batch(i) for i in range(start, end))

    def __len__(self):
        return self.num_batches
    
    def on_epoch_start(self):
        """
        Placeholder function that does nothing
        """
        pass


@dataset_ex.command
def create_data_loader(
    motif_config_path, batch_size, num_batches, input_length, seq_alphabet,
    bg_seq_freqs, motif_center_dist_bound, negative_ratio, revcomp,
    background_match_reject_prob, background_match_score_thresh, num_workers,
    data_seed, neg_motif_config_path=None, return_configs=False
):
    """
    Creates a PyTorch DataLoader object which iterates through batches of data.
    Arguments:
        `motif_config_path`: path to JSON file containing motif configurations
            to sample from
        `neg_motif_config_path`: path to JSON file containing motif
            configurations for negative examples; defaults to having negatives
            just be random backgrounds
        `return_configs`: if True, each batch also returns the offsets and
            configurations used to create simulated sequences
    """
    pos_seq_simulator = SeqSimulator(
        motif_config_path, input_length, seq_alphabet, bg_seq_freqs,
        motif_center_dist_bound, data_seed
    )

    if neg_motif_config_path:
        neg_seq_simulator = SeqSimulator(
            neg_motif_config_path, input_length, seq_alphabet, bg_seq_freqs,
            motif_center_dist_bound, data_seed
        )
    else:
        neg_seq_simulator = None


    dataset = SimulatedSeqDataset(
        pos_seq_simulator, batch_size, num_batches, negative_ratio,
        neg_seq_simulator, revcomp, background_match_reject_prob,
        background_match_score_thresh, return_configs=return_configs
    )

    generator = torch.Generator()
    if data_seed is not None:
        # This sets the initial state of torch.initial_seed(), making future
        # calls to it deterministic
        generator.manual_seed(data_seed)
    else:
        # This makes sure that when there is no seed provided, the generator is
        # seeded randomly
        generator.seed()

    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            seed = torch.initial_seed() % (2 ** 32)
            worker_info.dataset.pos_seq_simulator.set_seed(
                seed + (2 * worker_id)
            )
            if worker_info.dataset.neg_seq_simulator is not None:
                worker_info.dataset.neg_seq_simulator.set_seed(
                    seed + (2 * worker_id) + 1
                )

    # Dataset loader: dataset is iterable and already returns batches
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=None, num_workers=num_workers,
        collate_fn=lambda x: x, worker_init_fn=worker_init_fn,
        generator=generator
    )

    return loader


@dataset_ex.automain
def main():
    import tqdm
    from datetime import datetime

    print("Testing reproducibility of sequences after seeding")

    motif_config_path = "/projects/site/gred/resbioai/tsenga5/mechint_regnet/data/simulations/configs/tal_gata_sanit_mix_config.json"

    loader_with_seed_1 = create_data_loader(
        motif_config_path, input_length=100, motif_center_dist_bound=20,
        batch_size=10, num_batches=1, num_workers=2, data_seed=123
    )
    loader_with_seed_2 = create_data_loader(
        motif_config_path, input_length=100, motif_center_dist_bound=20,
        batch_size=10, num_batches=1, num_workers=2, data_seed=123
    )
    loader_no_seed_1 = create_data_loader(
        motif_config_path, input_length=100, motif_center_dist_bound=20,
        batch_size=10, num_batches=1, num_workers=2
    )
    loader_no_seed_2 = create_data_loader(
        motif_config_path, input_length=100, motif_center_dist_bound=20,
        batch_size=10, num_batches=1, num_workers=2
    )

    seqs_with_seed_1 = util.one_hot_to_seqs(next(iter(loader_with_seed_1))[0])
    seqs_with_seed_2 = util.one_hot_to_seqs(next(iter(loader_with_seed_2))[0])
    seqs_no_seed_1 = util.one_hot_to_seqs(next(iter(loader_no_seed_1))[0])
    seqs_no_seed_2 = util.one_hot_to_seqs(next(iter(loader_no_seed_2))[0])
    seqs_with_seed_1 = set(seqs_with_seed_1)
    seqs_with_seed_2 = set(seqs_with_seed_2)
    seqs_no_seed_1 = set(seqs_no_seed_1)
    seqs_no_seed_2 = set(seqs_no_seed_2)
    print("Same seed: Unique to 1 = %d, Unique to 2 = %d, Shared = %d" % (
        len(seqs_with_seed_1 - seqs_with_seed_2),
        len(seqs_with_seed_2 - seqs_with_seed_1),
        len(seqs_with_seed_1 & seqs_with_seed_2)
    ))
    print("No seed: Unique to 1 = %d, Unique to 2 = %d, Shared = %d" % (
        len(seqs_no_seed_1 - seqs_no_seed_2),
        len(seqs_no_seed_2 - seqs_no_seed_1),
        len(seqs_no_seed_1 & seqs_no_seed_2)
    ))
    
    print()
    print("Testing uniqueness of sequences across batches and epochs")

    motif_config_path = "/projects/site/gred/resbioai/tsenga5/mechint_regnet/data/simulations/configs/tal_gata_sanit_mix_config.json"
    num_epochs = 5

    loader = create_data_loader(
        motif_config_path, input_length=100, motif_center_dist_bound=20,
        batch_size=8, num_batches=100, num_workers=5
    )
    
    start_time = datetime.now()
    
    all_seqs = []
    for _ in range(num_epochs):
        loader.dataset.on_epoch_start()
        epoch_seqs = []
        for batch in tqdm.tqdm(loader):
            epoch_seqs.extend(util.one_hot_to_seqs(batch[0]))
        all_seqs.append(epoch_seqs)
    
    end_time = datetime.now()
    print("Time: %ds" % (end_time - start_time).seconds)

    print("\tUnique\tTotal")
    all_seqs_set = set()
    for epoch_i, epoch_seqs in enumerate(all_seqs):
        epoch_seqs_set = set(epoch_seqs)
        all_seqs_set.update(epoch_seqs_set)
        print("Epoch %d\t%d\t%d" % (
            epoch_i, len(epoch_seqs_set), len(epoch_seqs)
        ))
    print("All\t%d\t%d" % (
        len(all_seqs_set), sum(len(epoch_seqs) for epoch_seqs in all_seqs)
    ))

    print()
    print("Testing prevalence of mix of motifs in background sequences")

    motif_config_path = "/projects/site/gred/resbioai/tsenga5/mechint_regnet/data/simulations/configs/tal_gata_sanit_mix_config.json"

    loader = create_data_loader(
        motif_config_path, input_length=1000, num_batches=100, data_seed=123,
        background_match_reject_prob=1
    )
    loader.dataset.on_epoch_start()

    start_time = datetime.now()
   
    all_seqs, all_labels = [], []
    for batch in tqdm.tqdm(loader, total=len(loader.dataset)):
        one_hots, labels = batch
        all_seqs.extend(util.one_hot_to_seqs(one_hots))
        all_labels.extend(list(labels.astype(int)))

    end_time = datetime.now()
    print("Time: %ds" % (end_time - start_time).seconds)

    motifs = {
        "GATA": ["TGATAA", "AGATAA"],
        "TAL": ["CAGCTG", "CAGATG"]
    }
    rc_dict = {"A": "T", "T": "A", "C": "G", "G": "C"}
    motifs_with_rc = {
        key : motif_list + [
            "".join([rc_dict[c] for c in motif][::-1]) for motif in motif_list
        ] for key, motif_list in motifs.items()
    }
    motifs_with_rc = {
        key : list(set(motif_list))
        for key, motif_list in motifs_with_rc.items()
    }
    counts = [
        {key: [] for key in motifs.keys()},
        {key: [] for key in motifs.keys()}
    ]
    for i, seq in enumerate(all_seqs):
        for key, motif_list in motifs_with_rc.items():
            counts[all_labels[i]][key].append(
                sum(seq.count(motif) for motif in motif_list)
            )
    neg_counts = list(zip(counts[0]["GATA"], counts[0]["TAL"]))
    pos_counts = list(zip(counts[1]["GATA"], counts[1]["TAL"]))
    neg_counts_dict, pos_counts_dict = {}, {}
    for tup in neg_counts:
        try:
            neg_counts_dict[tup] += 1
        except KeyError:
            neg_counts_dict[tup] = 1
    for tup in pos_counts:
        try:
            pos_counts_dict[tup] += 1
        except KeyError:
            pos_counts_dict[tup] = 1
    print("Positive-sequence GATA-TAL motif counts")
    for tup in sorted(pos_counts_dict, key=(lambda k: -pos_counts_dict[k])):
        print("\t%d-%d\t%d" % (tup + (pos_counts_dict[tup],)))
    print("Negative-sequence GATA-TAL motif counts")
    for tup in sorted(neg_counts_dict, key=(lambda k: -neg_counts_dict[k])):
        print("\t%d-%d\t%d" % (tup + (neg_counts_dict[tup],)))
    
    print()
    print("Testing distribution of motif placement in positive sequences")

    motif_config_path = "/projects/site/gred/resbioai/tsenga5/mechint_regnet/data/simulations/configs/ctcf_sanit_soft_spacing_config.json"
    
    loader = create_data_loader(
        motif_config_path, input_length=50, motif_center_dist_bound=5,
        batch_size=1000, num_batches=20, data_seed=123
    )
    loader.dataset.on_epoch_start()

    start_time = datetime.now()
   
    all_seqs = set()
    for batch in tqdm.tqdm(loader, total=len(loader.dataset)):
        one_hots, labels = batch
        seqs = util.one_hot_to_seqs(one_hots)
        for seq in seqs:
            assert seq not in all_seqs
            all_seqs.add(seq)

    end_time = datetime.now()
    print("Time: %ds" % (end_time - start_time).seconds)

    k = 2
    rc_k = (len(labels) // 2) + k
    
    print("Example sequence and reverse complement:")
    print(seqs[k][:20] + "..." + seqs[k][-20:])
    print(seqs[rc_k][:20] + "..." + seqs[rc_k][-20:])

    print()
   
    ctcf_motif = "CCACCAGGGGG"
    ctcf_rc_motif = "CCCCCTGGTGG"
    pos_inds = []
    for i in range(len(seqs)):
        inds = [
            j for j in range(len(seqs[i]))
            if seqs[i][j:j+len(ctcf_motif)] == ctcf_motif
        ]
        inds_rc = [
            j for j in range(len(seqs[i]))
            if seqs[i][j:j+len(ctcf_motif)] == ctcf_rc_motif
        ]
        if i < len(seqs) / 2 and labels[i] == 1:
            assert inds and not inds_rc
            pos_inds.append(inds)
        elif i >= len(seqs) / 2 and labels[i] == 1:
            assert inds_rc and not inds
            pos_inds.append(inds_rc)
        else:
            assert not inds and not inds_rc

    assert all(len(p) in (1, 2) for p in pos_inds)
    print("Single-motif instances: %d" % sum(len(p) == 1 for p in pos_inds))
    print("Double-motif instances: %d" % sum(len(p) == 2 for p in pos_inds))
    print("\tGap distribution:")
    gaps = [p[1] - p[0] - len(ctcf_motif) for p in pos_inds if len(p) == 2]
    vals, counts = np.unique(gaps, return_counts=True)
    for v, c in zip(vals, counts):
        print("\t%d: %d" % (v, c))
    
    print("Indices of first motif hits, in batch order:")
    print(" ".join([str(min(p)) for p in pos_inds]))

    print()
    print("Testing returned configurations")

    motif_config_path = "/projects/site/gred/resbioai/tsenga5/mechint_regnet/data/simulations/configs/ctcf_sanit_soft_spacing_config.json"

    loader = create_data_loader(
        motif_config_path, input_length=50, motif_center_dist_bound=5,
        batch_size=64, num_batches=20, data_seed=123, return_configs=True
    )
    loader.dataset.on_epoch_start()
    
    start_time = datetime.now()

    all_seqs, all_labels, all_configs = [], [], []
    for batch in tqdm.tqdm(loader, total=len(loader.dataset)):
        one_hots, labels, configs = batch
        seqs = util.one_hot_to_seqs(one_hots)
        all_seqs.append(seqs)
        all_labels.append(labels)
        all_configs.append(configs)

    all_seqs = sum(all_seqs, [])
    all_labels = np.concatenate(all_labels)
    all_configs = sum(all_configs, [])

    end_time = datetime.now()
    print("Time: %ds" % (end_time - start_time).seconds)

    ctcf_motif = "CCACCAGGGGG"
    ctcf_rc_motif = "CCCCCTGGTGG"
    
    num_nonempty_configs = 0
    for i in range(len(all_seqs)):
        if all_configs[i] is not None:
            start_pos, config = all_configs[i]
            seq = all_seqs[i]

            # Forward orientation
            s, forward_subseqs = start_pos, []
            for token in config:
                if token == "CTCF_sanit":
                    forward_subseqs.append(seq[s : s + len(ctcf_motif)])
                    s += len(ctcf_motif)
                elif type(token) is int:
                    s += token
                else:
                    raise ValueError("Unknown configuration")

            # Reverse orientation
            e, reverse_subseqs = len(seq) - start_pos, []
            for token in config[::-1]:
                if token == "CTCF_sanit":
                    reverse_subseqs.append(seq[e - len(ctcf_motif) : e])
                    e -= len(ctcf_motif)
                elif type(token) is int:
                    e -= token
                else:
                    raise ValueError("Unknown configuration")
            
            num_nonempty_configs += 1

            assert all(subseq == ctcf_motif for subseq in forward_subseqs) or \
                all(subseq == ctcf_rc_motif for subseq in reverse_subseqs)

    print("Number of sequences with configurations: %d" % num_nonempty_configs)
    print("Number of sequences without configurations: %d" % (
        len(all_configs) - num_nonempty_configs)
    )
