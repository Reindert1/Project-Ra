rule train_model:
    input:
        data_location =  config["dataset_dir"] + "dataset/full_classification.npy"
    output:
        trained_model = config["results_dir"] + "model/trained_model.h5",
        history_location= config["results_dir"] + "model/model_history.csv"

    params:
        early_stopping = config["early_stopping"]

    script:
        "../scripts/deep_learning/train_model.py"