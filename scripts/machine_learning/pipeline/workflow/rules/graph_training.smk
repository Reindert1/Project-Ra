rule graph_training:
    input:
        history_location = config["model_history"]

    output:
        accuracy = config["graph_accuracy"],
        loss = config["graph_loss"],

    script:
        "../scripts/graph_training.py"