rule train_test_split:
    input:
        config["dataset_dir"] + "dataset/full_classification.npy"
    output:
        config["dataset_dir"] + "dataset/train.h5py"
    script:
        "../scripts/train_test_split.py"

rule sgd_classifier:
    input:
        config["dataset_dir"] + "dataset/train.h5py"
    output:
        config["dataset_dir"] + "models/SGD.sav"
    script:
        "../scripts/sgd_classifier.py"

rule gaussian_nb:
    input:
        config["dataset_dir"] + "dataset/train.h5py"
    output:
        config["dataset_dir"] + "models/GaussianNB.sav"
    script:
        "../scripts/gaussiannb.py"

rule zero_r:
    input:
        config["dataset_dir"] + "dataset/train.h5py"
    output:
        config["dataset_dir"] + "models/ZeroR.sav"
    script:
        "../scripts/zero_r.py"