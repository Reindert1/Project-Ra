rule train_test_split:
    input:
        config["dataset_dir"] + "dataset/full_classification.npy"
    output:
        config["dataset_dir"] + "dataset/train.h5py"
    threads:
        1
    message:
        "Splitting training data into test and train dataset"
    log:
        notebook=config["results_dir"] + "logs/train_test_split/train_test_split.log"
    script:
        "../scripts/machine_learning/train_test_split.py"

rule sgd_classifier:
    input:
        config["dataset_dir"] + "dataset/train.h5py"
    output:
        model=config["dataset_dir"] + "models/SGD.sav",
        metrics=config["dataset_dir"] + "model_metrics/SGD.sav"
    threads:
        1
    message:
        "Training SGD classifier"
    log:
        notebook=config["results_dir"] + "logs/models/sgd_classifier.log"
    script:
        "../scripts/machine_learning/sgd_classifier.py"

rule gaussian_nb:
    input:
        config["dataset_dir"] + "dataset/train.h5py"
    output:
        model=config["dataset_dir"] + "models/GaussianNB.sav",
        metrics=config["dataset_dir"] + "model_metrics/GaussianNB.sav"
    threads:
        1
    message:
        "Training Gaussian Naive Bayes classifier"
    log:
        notebook=config["results_dir"] + "logs/models/gaussian_nb.log"
    script:
        "../scripts/machine_learning/gaussiannb.py"

rule zero_r:
    input:
        config["dataset_dir"] + "dataset/train.h5py"
    output:
        model=config["dataset_dir"] + "models/ZeroR.sav",
        metrics=config["dataset_dir"] + "model_metrics/ZeroR.sav"
    threads:
        1
    message:
        "Training ZeroR classifier"
    log:
        notebook=config["results_dir"] + "logs/models/zero_r.log"
    script:
        "../scripts/machine_learning/zero_r.py"

rule multinomial_nb:
    input:
        config["dataset_dir"] + "dataset/train.h5py"
    output:
        model=config["dataset_dir"] + "models/MultinomialNB.sav",
        metrics=config["dataset_dir"] + "model_metrics/MultinomialNB.sav"
    threads:
        1
    message:
        "Training Multinomial Naive Bayes classifier"
    log:
        notebook=config["results_dir"] + "logs/models/multinomial_nb.log"
    script:
        "../scripts/machine_learning/multinomialnb.py"