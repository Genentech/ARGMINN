install-dependencies:
	conda install -y -c conda-forge click scipy numpy pymongo scikit-learn pandas jupyter tqdm matplotlib h5py
	pip install sacred tables vdom logomaker cmake
	pip install modisco-lite
	conda install -c pytorch -c nvidia pytorch torchvision captum pytorch-cuda=12.1
	conda install -y -c bioconda pyfaidx
	pip install pysam
