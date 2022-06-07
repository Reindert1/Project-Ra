rule train_model:
    input:
        data_location = config["input_data"],
    output:
        trained_model = config["trained_model"],
        history_location= config["model_history"]

    params:
        early_stopping = config["early_stopping"]

    script:
        "../scripts/train_model.py"