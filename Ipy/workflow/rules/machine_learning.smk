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