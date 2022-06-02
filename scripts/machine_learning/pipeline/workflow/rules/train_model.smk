rule train_model:
    input:
        data_location = config["input_data"],
        early_stopping = config["early_stopping"]
    output:
        config["model_export"] + "trained_model.h5"

    script:
        "../scripts/train_model.py"