rule graph_training:
    input:
        history_location = config["results_dir"] + "model/model_history.csv"

    output:
        accuracy = config["results_dir"] + "graphs/loss.png",
        loss = config["results_dir"] + "graphs/accuracy.png"

    script:
        "../scripts/deep_learning/graph_training.py"