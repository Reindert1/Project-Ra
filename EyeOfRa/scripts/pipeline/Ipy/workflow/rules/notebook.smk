def rem_nn():
    algs = config["algorithms"].copy()
    if "NN" in algs:
        algs.remove("NN")
    return algs

def get_graphs():
    algs = config["algorithms"].copy()
    if "NN" in algs:
        graphs = [config["results_dir"] + "graphs/loss.png",  config["results_dir"] + "graphs/accuracy.png"]
    else:
        graphs = []
    return graphs

rule create_output_notebook:
    input:
        images=expand(config["results_dir"] + "images/overlayed/{classifier}_{model_name}_overlay.tif",
                      classifier=config["segment"], model_name=config["algorithms"]),
        metrics=expand(config["results_dir"] + "model_metrics/{model_name}.sav",
                       model_name=rem_nn()),
        graphs=get_graphs(),
        reports=expand(config["results_dir"] + "model_metrics/{model_name}_report.sav",
                       model_name=rem_nn())
    output:
        temp(touch(config["results_dir"] + "notebook.done"))
    threads:
        1
    message:
        "Building final notebook"
    log:
        notebook=config["results_dir"] + "notebooks/output_notebook.ipynb"
    benchmark:
        config["results_dir"] + "benchmarks/notebooks/output_notebook.benchmark.txt"
    notebook:
        "../notebooks/output_notebook.py.ipynb"