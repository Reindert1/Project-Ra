rule create_output_notebook:
    input:
        images=expand(config["results_dir"] + "images/overlayed/{classifier}_{model_name}_overlay.tif",
                      classifier=config["segment"], model_name=config["algorithms"]),
        metrics=expand(config["dataset_dir"] + "model_metrics/{model_name}.sav",
                       model_name=config["algorithms"])
    output:
        temp(config["results_dir"] + "notebook.done")
    log:
        notebook=config["results_dir"] + "notebooks/output_notebook.ipynb"
    notebook:
        "../notebooks/output_notebook.py.ipynb"