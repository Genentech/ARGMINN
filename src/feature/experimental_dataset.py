import torch
import numpy as np
import pandas as pd
import pysam
import sacred
import feature.util as util

dataset_ex = sacred.Experiment("exp_dataset")

@dataset_ex.config
def config():
    # Path to genome Fasta
    genome_fasta_path = "/home/tsenga5/mechint_regnet/data/genomes/hg38.fa"

    # Path to chromosome sizes
    chrom_sizes_path = "/home/tsenga5/mechint_regnet/data/genomes/GRCh38_EBV.chrom.sizes.tsv"

    # Size of batches
    batch_size = 128

    # For each input sequence in the raw data, center it and pad to this length 
    input_length = 500

    # Central region of the sequence to overlap a peak to be considered positive
    input_center_overlap = 400

    # Sample this many negatives randomly for every positive example
    negative_ratio = 1

    # Whether or not to perform reverse complement augmentation
    revcomp = True

    # Number of workers for the data loader
    num_workers = 10
    
    # Seed for generating examples
    data_seed = None


class PeakCoordSampler:
    def __init__(
        self, peaks_bed_path, coord_size, coord_center_size, data_seed,
        overlap_fraction=0.5, chrom_set=None
    ):
        """
        Allows for sampling coordinates from a peak BED file. Also allows for
        checking if a coordinate overlaps a peak.
        Arguments:
            `peaks_bed_path`: path to BED file (which may be gzipped) in
                NarrowPeak format
            `input_size`: length of coordinates to generate
            `coord_center_size`: central length of a coordinate to consider for
                overlaps with peaks; if this is -1, then overlap must be an
                exact match to the peak summit
            `data_seed`: seed for sampling coordinates
            `overlap_fraction`: fraction of region overlap required
            `chrom_set`: if given, an iterable of chromosomes to sample from;
                otherwise samples from all available peaks
        """
        peaks_table = util.import_peaks_bed(peaks_bed_path)
        self.coord_size = coord_size
        self.coord_center_size = coord_center_size 
        
        self.rng = np.random.default_rng(data_seed)

        # Limit chromosomes if needed
        if chrom_set:
            peaks_table = peaks_table[peaks_table["chrom"].isin(chrom_set)]

        peaks_table["summit"] = \
            peaks_table["peak_start"] + peaks_table["summit_offset"]
        peaks_table["min_overlap_size"] = \
            (overlap_fraction * np.minimum(
                peaks_table["peak_start"] - peaks_table["peak_end"],
                coord_center_size 
            )).astype(int)  # min(PF, QF) (see below)
        self.peaks_table = peaks_table
    
    def set_seed(self, data_seed):
        """
        Sets seed of RNG.
        Arguments:
            `data_seed`: integer-valued seed
        """
        self.rng = np.random.default_rng(data_seed)

    def get_peak_overlapping_coord(self, index):
        """
        Returns a coordinate which overlaps the peak at the given index. The
        returned coordinate will be randomized so long as at least a sufficient
        fraction of the central region overlaps the peak, or at least a fraction
        of the peak overlaps the central region (whichever is smaller).
        Arguments:
            `index`: index of peak for coordinate to return
        Returns a coordinate a a triplet of chromosome, start position, and end
        positions. The coordinate is length `coord_size`.
        """
        peak_row = self.peaks_table.iloc[index]

        if self.coord_center_size == -1:
            # Just return a summit-centered coordinate
            chrom, summit = peak_row[["chrom", "summit"]]
            coord_start = summit - (self.coord_size // 2)
            coord_end = coord_start + self.coord_size
            return chrom, coord_start, coord_end

        # Let the query region coordinates be [Q1, Q2] (Q2 - Q1 = Q), and the
        # peak region be [P1, P2] (P2 - P1 = P). The query region is the region
        # we are considering for an overlap within the query.
        # If P = Q, then we require Q2 - P1 >= QF AND P2 - Q1 >= QF.
        # If P < Q, then we require Q2 - P1 >= PF AND P2 - Q1 >= PF.
        # If P > Q, then we require Q2 - P1 >= QF AND P2 - Q1 >= QF.
        # In general, if we let M = min(QF, PF), then we require:
        # Q2 - P1 >= M AND P2 - Q1 >= M. Substituting Q2 = Q1 + Q, we have that
        # Q1 must be in [P1 - Q + M, P2 - M].
        chrom, peak_start, peak_end, min_overlap_size = \
            peak_row[["chrom", "peak_start", "peak_end", "min_overlap_size"]]
        q1 = self.rng.integers(
            peak_start - self.coord_center_size + min_overlap_size,
            peak_end - min_overlap_size + 1
        )
        coord_start = q1 - \
            ((self.coord_size - self.coord_center_size) // 2)
        coord_end = coord_start + self.coord_size
        return chrom, coord_start, coord_end

    def check_coord_overlap(self, chrom, start, end, return_peaks=False):
        """
        Given a query coordinate, checks if the central region (of length
        `coord_center_size`) centered at the same center as the query coordinate
        sufficiently overlaps with any peak. Note that the query center region
        will be extended (or shortened) as needed based on the center of the
        given coordinate. The query center region is considered to overlap with
        a peak if at least a sufficient fraction of the center region covers the
        peak, or at least a sufficient fraction of the peak covers the query
        center region (if the peak is smaller).
        Arguments:
            `chrom`: chromosome of the query coordinate (e.g. "chr1")
            `start`: start coordinate of the query
            `end`: end coordinate of the query
            `return_peaks`: if True, also return the overlapping peaks
        Returns a boolean for whether or not the query coordinate overlaps any
        peak. If `return_peak` is True, then also returns a Pandas Dataframe
        subtable of the peaks which overlapped with the query coordinate. This
        subtable will be empty if no peak overlapped.
        """
        q1 = ((end - start) // 2) - (self.coord_center_size // 2)
        q2 = q1 + self.coord_center_size
        
        # First subset to the right chromosome
        match_table = self.peaks_table[self.peaks_table["chrom"] == chrom]
        
        # Now check for sufficient overlaps
        min_overlap_size = match_table["min_overlap_size"]
        match_table = match_table[
            ((match_table["peak_end"] - q1) >= min_overlap_size) &
            ((q2 - match_table["peak_start"]) >= min_overlap_size)
        ]

        if return_peaks:
            return not match_table.empty, match_table
        return not match_table.empty

    def __len__(self):
        """
        Returns the length of the peak set.
        """
        return len(self.peaks_table)


class PeakSeqDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, peak_coord_sampler, genome_fasta_path, chrom_sizes_path,
        batch_size, input_length, negative_ratio, data_seed, revcomp=False,
        chrom_set=None, return_coords=False
    ):
        """
        Generates batches of one-hot-encoded sequences and binary labels based
        on a set of peaks.
        Arguments:
            `peak_coord_sampler (PeakCoordSampler): generates genomic
                coordinates that overlap with peaks
            `genome_fasta_path`: path to genome Fasta
            `chrom_sizes_path`: path to TSV of chromosomes and their total sizes
            `batch_size`: number of sequences per batch, B
            `input_length`: length of sequences to generate, L
            `negative_ratio`: generate this many negative sequences per batch as
                positive ones
            `data_seed`: seed for sampling coordinates
            `revcomp`: whether or not to perform revcomp to the batch; this will
                not change the batch size, but halve the number of unique
                objects per batch; if True, `batch_size` must be even
            `chrom_set`: if given, an iterable of chromosomes to sample from;
                otherwise samples from all available chromosomes
            `return_coords`: if True, also returns coordinates in each batch
                along with labels
        Returns a B x L x D NumPy array of one-hot encodings and a B-array of 
        binary labels. If `return_coords` is True, also returns a B x 3 object
        array of coordinates for each sequence.
        """
        self.peak_coord_sampler = peak_coord_sampler
        self.genome_fasta_path = genome_fasta_path
        self.batch_size = batch_size
        self.input_length = input_length
        self.negative_ratio = negative_ratio
        self.revcomp = revcomp
        self.return_coords = return_coords

        if revcomp:
            assert batch_size % 2 == 0
        pre_revcomp_size = batch_size // (2 if revcomp else 1)

        self.num_pos_per_batch = int(
            np.ceil(pre_revcomp_size / (1 + negative_ratio))
        )
        self.num_neg_per_batch = pre_revcomp_size - self.num_pos_per_batch
        self.num_batches = int(
            np.ceil(len(peak_coord_sampler) / self.num_pos_per_batch)
        )
         
        # Import table of chromosome sizes
        chrom_sizes_table = pd.read_csv(
            chrom_sizes_path, sep="\t", header=None, names=["chrom", "max_size"]
        )
        # Limit chromosomes if needed
        if chrom_set:
            chrom_sizes_table = chrom_sizes_table[
	        chrom_sizes_table["chrom"].isin(chrom_set)
	    ]
        # Cut off max sizes to avoid overrunning ends of chromosome
        chrom_sizes_table["max_size"] -= input_length
        # Compute sampling weights
        chrom_sizes_table["weight"] = \
            chrom_sizes_table["max_size"] / chrom_sizes_table["max_size"].sum()
        self.chrom_sizes_table = chrom_sizes_table

        self.rng = np.random.default_rng(data_seed)
    
    def set_seed(self, data_seed):
        """
        Sets seed of RNG.
        Arguments:
            `data_seed`: integer-valued seed
        """
        self.rng = np.random.default_rng(data_seed)

    def _get_random_coord(self):
        """
        Randomly samples a coordinate of length `input_length` from the genome.
        Returns a coordinate as a triplet of chromosome, start, and end.
        """
        chrom_sample = self.chrom_sizes_table.sample(
            n=1,
            weights=self.chrom_sizes_table["weight"],
            random_state=self.rng
        ).iloc[0]
        start = (self.rng.random() * chrom_sample["max_size"]).astype(int)
        end = start + self.input_length
        return chrom_sample["chrom"], start, end

    def get_batch(self, index):
        """
        Returns a batch, which consists of a B x L x D NumPy array of 1-hot
        encoded sequences and a B-array of labels.
        Arguments:
            `index`: index of batch to return, for positive sequences
        """
        assert index < self.num_batches
        
        batch_start = index * self.num_pos_per_batch
        batch_end = min(
            batch_start + self.num_pos_per_batch, len(self.peak_coord_inds)
        )
        pos_inds = self.peak_coord_inds[batch_start : batch_end]
        num_pos = len(pos_inds)

        # Sample coordinates overlying peaks
        pos_coords = np.array([
            self.peak_coord_sampler.get_peak_overlapping_coord(i)
            for i in pos_inds
        ], dtype=object)

        # Sample random coordinates
        if num_pos == self.num_pos_per_batch:
            num_neg = self.num_neg_per_batch
        else:
            # Adjust the number of negatives in the batch to retain the desired
            # ratio; for the last batch
            num_neg = int(
                self.num_neg_per_batch / self.num_pos_per_batch * num_pos
            )
        neg_coords = np.array([
            self._get_random_coord() for _ in range(num_neg)
        ], dtype=object)
        if not neg_coords.size:
            neg_coords = np.empty((0, 3), dtype=object)
        
        pos_labels, neg_labels = np.ones(num_pos), np.zeros(num_neg)
        # Adjust the label of negative coordinates if any overlie a peak
        for i, coord in enumerate(neg_coords):
            if self.peak_coord_sampler.check_coord_overlap(
                coord[0], coord[1], coord[2]
            ):
                neg_labels[i] = 1
        labels = np.concatenate([pos_labels, neg_labels])

        # Convert the coordinates into one-hot sequences
        # Create FASTA reader here to avoid parallelism issues
        genome_reader = pysam.FastaFile(self.genome_fasta_path)
        coords = np.concatenate([pos_coords, neg_coords])
        seqs = [
            genome_reader.fetch(chrom, max(0, start), end)
            for chrom, start, end in coords
        ]

        # If any of the sequences are too short, pad with Ns on either side
        for i, seq in enumerate(seqs):
            if len(seq) != self.input_length:
                diff = self.input_length - len(seq)
                left_pad = diff // 2
                right_pad = diff - left_pad
                seqs[i] = (left_pad * "N") + seq + (right_pad * "N")

        one_hots = util.seqs_to_one_hot(seqs)
        
        if self.revcomp:
            one_hots = np.concatenate([
                one_hots, np.flip(one_hots, axis=(1, 2))
            ])
            labels = np.concatenate([labels, labels])
            if self.return_coords:
                coords = np.concatenate([coords, coords])
        
        if self.return_coords:
            return one_hots, labels, coords
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
        Randomizes the ordering of positive sequences to fetch. Must be called
        before iterating.
        """
        self.peak_coord_inds = self.rng.permutation(
            len(self.peak_coord_sampler)
        )


@dataset_ex.command
def create_data_loader(
    peaks_bed_path, genome_fasta_path, chrom_sizes_path, batch_size,
    input_length, input_center_overlap, negative_ratio, revcomp, num_workers,
    data_seed, chrom_set=None, return_coords=False
):
    """
    Creates a PyTorch DataLoader object which iterates through batches of data.
    Arguments:
        `peaks_bed_path`: path to BED file (which may be gzipped) in NarrowPeak
            format
        `chrom_set`: if given, an iterable of chromosomes to sample from;
            otherwise samples from all available peaks
        `return_coords`: if True, also returns coordinates in each batch along
            with labels
    """
    peak_coord_sampler = PeakCoordSampler(
        peaks_bed_path, input_length, input_center_overlap, data_seed,
        chrom_set=chrom_set
    )
    
    dataset = PeakSeqDataset(
        peak_coord_sampler, genome_fasta_path, chrom_sizes_path, batch_size,
        input_length, negative_ratio, data_seed, revcomp=revcomp,
        chrom_set=chrom_set, return_coords=return_coords
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
            worker_info.dataset.set_seed(
                seed + (2 * worker_id)
            )
            worker_info.dataset.peak_coord_sampler.set_seed(
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
    global data, loader
    import tqdm
    from datetime import datetime
    import scipy.signal
    import scipy.stats
    import analysis.motif_util as motif_util

    peaks_bed_path = "/home/tsenga5/mechint_regnet/data/encode/chipseq/ENCSR607XFI_CTCF_HepG2/ENCFF664UGR_idrpeaks.bed.gz"
    full_chrom_set = ["chr" + str(i) for i in range(1, 24)] + ["chrX"]
    
    print("Testing correctness of coordinates from genome")
   
    num_batches = 20

    loader = create_data_loader(
        peaks_bed_path, batch_size=100, data_seed=123, chrom_set=full_chrom_set,
        return_coords=True, num_workers=5
    )

    loader.dataset.on_epoch_start()
    loader_iter = iter(loader)
    all_seqs, all_rc_seqs, all_coords, all_rc_coords = [], [], [], []
    for _ in tqdm.trange(num_batches):
        one_hots, _, coords = next(loader_iter)
        seqs = util.one_hot_to_seqs(one_hots)
        coords = [tuple(c) for c in coords]
        # Split into forward and reverse complement
        batch_size = len(seqs)
        mid = batch_size // 2
        all_seqs.extend(seqs[:mid])
        all_rc_seqs.extend(seqs[mid:])
        all_coords.extend(coords[:mid])
        all_rc_coords.extend(coords[mid:])
    
    # Check coordinates are the same in forward and reverse complement
    assert all_coords == all_rc_coords

    # Check reverse-complement sequences are correct relative to forward
    rc_dict = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    for i, rc_seq in enumerate(all_rc_seqs):
        assert all_seqs[i] == "".join([rc_dict[c] for c in rc_seq][::-1])

    # Check forward sequences are correct for each coordinate
    genome_fasta_path = loader.dataset.genome_fasta_path
    genome_reader = pysam.FastaFile(genome_fasta_path)
    for i, coord in enumerate(all_coords):
        assert genome_reader.fetch(*coord).upper() == all_seqs[i]
    
    print()
    print("Testing reproducibility of sequences after seeding")

    loader_with_seed_1 = create_data_loader(
        peaks_bed_path, batch_size=50, data_seed=123, chrom_set=["chr1"],
        return_coords=True, num_workers=2
    )
    loader_with_seed_2 = create_data_loader(
        peaks_bed_path, batch_size=50, data_seed=123, chrom_set=["chr1"],
        return_coords=True, num_workers=2
    )
    loader_no_seed_1 = create_data_loader(
        peaks_bed_path, batch_size=50, chrom_set=["chr1"], return_coords=True,
        num_workers=2
    )
    loader_no_seed_2 = create_data_loader(
        peaks_bed_path, batch_size=50, chrom_set=["chr1"], return_coords=True,
        num_workers=2
    )

    def get_one_batch_seqs_coords(loader):
        loader.dataset.on_epoch_start()
        batch = next(iter(loader))
        seqs, labels, coords = batch
        return set(util.one_hot_to_seqs(seqs)), set([tuple(c) for c in coords])

    seqs_with_seed_1, coords_with_seed_1 = \
        get_one_batch_seqs_coords(loader_with_seed_1)
    seqs_with_seed_2, coords_with_seed_2 = \
        get_one_batch_seqs_coords(loader_with_seed_2)
    seqs_no_seed_1, coords_no_seed_1 = \
        get_one_batch_seqs_coords(loader_no_seed_1)
    seqs_no_seed_2, coords_no_seed_2 = \
        get_one_batch_seqs_coords(loader_no_seed_2)
    
    print("Uniqueness by sequence:")
    print("\tSame seed: Unique to 1 = %d, Unique to 2 = %d, Shared = %d" % (
        len(seqs_with_seed_1 - seqs_with_seed_2),
        len(seqs_with_seed_2 - seqs_with_seed_1),
        len(seqs_with_seed_1 & seqs_with_seed_2)
    ))
    print("\tNo seed: Unique to 1 = %d, Unique to 2 = %d, Shared = %d" % (
        len(seqs_no_seed_1 - seqs_no_seed_2),
        len(seqs_no_seed_2 - seqs_no_seed_1),
        len(seqs_no_seed_1 & seqs_no_seed_2)
    ))
    print("Uniqueness by coordinate:")
    print("\tSame seed: Unique to 1 = %d, Unique to 2 = %d, Shared = %d" % (
        len(coords_with_seed_1 - coords_with_seed_2),
        len(coords_with_seed_2 - coords_with_seed_1),
        len(coords_with_seed_1 & coords_with_seed_2)
    ))
    print("\tNo seed: Unique to 1 = %d, Unique to 2 = %d, Shared = %d" % (
        len(coords_no_seed_1 - coords_no_seed_2),
        len(coords_no_seed_2 - coords_no_seed_1),
        len(coords_no_seed_1 & coords_no_seed_2)
    ))

    print()
    print("Testing uniqueness of sequences across batches and epochs (with revcomp)")
    
    num_epochs = 5

    loader = create_data_loader(
        peaks_bed_path, batch_size=8, data_seed=123, chrom_set=["chr21"],
        return_coords=True, num_workers=5
    )
    
    start_time = datetime.now()
    
    all_labels, all_coords = [], []
    for _ in range(num_epochs):
        loader.dataset.on_epoch_start()
        epoch_labels, epoch_coords = [], []
        for batch in tqdm.tqdm(loader):
            epoch_labels.extend(list(batch[1]))
            epoch_coords.extend([tuple(c) for c in batch[2]])
        all_labels.append(np.array(epoch_labels))
        all_coords.append(epoch_coords)
    
    end_time = datetime.now()
    print("Time: %ds" % (end_time - start_time).seconds)

    print("\tUnique+\tTotal+\tUnique-\tTotal-")
    all_pos_coords, all_neg_coords = set(), set()
    for epoch_i in range(num_epochs):
        pos_inds = np.where(all_labels[epoch_i] == 1)[0]
        neg_inds = np.where(all_labels[epoch_i] == 0)[0]
        pos_coords = set([all_coords[epoch_i][i] for i in pos_inds])
        neg_coords = set([all_coords[epoch_i][i] for i in neg_inds])
        all_pos_coords.update(pos_coords)
        all_neg_coords.update(neg_coords)
        print("Epoch %d\t%d\t%d\t%d\t%d" % (
            epoch_i, len(pos_coords), len(pos_inds), len(neg_coords),
            len(neg_inds)
        ))
    print("All\t%d\t%d\t%d\t%d" % (
        len(all_pos_coords), sum(np.sum(labels == 1) for labels in all_labels),
        len(all_neg_coords), sum(np.sum(labels == 0) for labels in all_labels)
    ))
    print("Total unique coordinates: %d" % len(all_pos_coords | all_neg_coords))
    print("Total coordinates: %d" % sum(len(labels) for labels in all_labels))

    print()
    print("Testing presence of motifs in positives")

    motif_file_path = "/projects/site/gred/resbioai/tsenga5/mechint_regnet/data/simulations/motifs/ctcf.txt"

    motif_dict = motif_util.import_meme_motifs([motif_file_path])
    ctcf_pfm = motif_dict["CTCF_exp"]
    ctcf_pfm_rc = np.flip(ctcf_pfm)

    loader = create_data_loader(
        peaks_bed_path, data_seed=123, chrom_set=["chr1"], num_workers=5
    )

    loader.dataset.on_epoch_start()

    start_time = datetime.now()
   
    all_one_hots, all_labels = [], []
    for batch in tqdm.tqdm(loader, total=len(loader.dataset)):
        all_one_hots.append(batch[0])
        all_labels.append(batch[1].astype(int))
    all_one_hots = np.concatenate(all_one_hots)
    all_labels = np.concatenate(all_labels)

    end_time = datetime.now()
    print("Time: %ds" % (end_time - start_time).seconds)

    max_match_scores = [[], []]
    for i in tqdm.trange(len(all_one_hots)):
        one_hot = all_one_hots[i]
        match_scores = scipy.signal.correlate(one_hot, ctcf_pfm, mode="valid")
        match_scores_rc = scipy.signal.correlate(
            one_hot, ctcf_pfm_rc, mode="valid"
        )
        # Shape: L - L' + 1 x 1
        max_match_scores[all_labels[i]].append(
            max(np.max(match_scores), np.max(match_scores_rc))
        )
    
    print("Average max match score in positives: %.2f" % np.mean(
        max_match_scores[1]
    ))
    print("Average max match score in negatives: %.2f" % np.mean(
        max_match_scores[0]
    ))
    print("t-test: p = %s" % scipy.stats.ttest_ind(
        max_match_scores[1], max_match_scores[0], alternative="greater"
    )[1])
