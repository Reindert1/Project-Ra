rule train_test_split:
    input:
        config["dataset_dir"] + "dataset/full_classification.npy"
    output:
        config["dataset_dir"] + "dataset/train.h5py"
    message:
        "Splitting training data into test and train dataset"
    log:
        notebook=config["results_dir"] + "logs/train_test_split/train_test_split.log"
    script:
        "../scripts/train_test_split.py"

rule sgd_classifier:
    input:
        config["dataset_dir"] + "dataset/train.h5py"
    output:
        model=config["dataset_dir"] + "models/SGD.sav",
        metrics=config["dataset_dir"] + "model_metrics/SGD.sav"
    message:
        "Training SGD classifier"
    log:
        notebook=config["results_dir"] + "logs/models/sgd_classifier.log"
    script:
        "../scripts/sgd_classifier.py"

rule gaussian_nb:
    input:
        config["dataset_dir"] + "dataset/train.h5py"
    output:
        model=config["dataset_dir"] + "models/GaussianNB.sav",
        metrics=config["dataset_dir"] + "model_metrics/GaussianNB.sav"
    message:
        "Training Gaussian Naive Bayes classifier"
    log:
        notebook=config["results_dir"] + "logs/models/gaussian_nb.log"
    script:
        "../scripts/gaussiannb.py"

rule zero_r:
    input:
        config["dataset_dir"] + "dataset/train.h5py"
    output:
        model=config["dataset_dir"] + "models/ZeroR.sav",
        metrics=config["dataset_dir"] + "model_metrics/ZeroR.sav"
    message:
        "Training ZeroR classifier"
    log:
        notebook=config["results_dir"] + "logs/models/zero_r.log"
    script:
        "../scripts/zero_r.py"