import os
import shutil

import yaml
import subprocess
import pathlib


class PipeLineManager:
    pipeline_path = "Ipy"
    storage = "warehouse/"
    options = {}

    def __init__(self):
        self.set_default_options()

    def run_full(self):
        removed_dirs = [
            "datasets",
            "results"
        ]

        workdir = pathlib.Path(self.storage)
        for p in removed_dirs:
            pat = pathlib.Path(p)
            fullpath = workdir/pat
            if fullpath.exists():
                shutil.rmtree(fullpath)

        self.run_snakemake(4)


    def train_only(self, model):
        pass

    def classify_only(self, model):
        pass

    def yaml_constructor(self, path):
        # Results --> results
        # Datasets --> datadir can same
        config = self.options
        with open(path, "w") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    def set_default_options(self):
        self.options["datadir"] = "/Users/sanderbouwman/School/Thema11/Themaopdracht/Project-Ra/Project-Ra/EyeOfRa/scripts/pipeline/warehouse/"
        self.options["train_data"] = "larger_data.tif"
        self.options["classifiers"] = {"vertical": "mask_larger_data.tif"}
        self.options["algorithms"] = ["NN"]
        self.options["gaussian_layers"] = 6
        self.options["window_size"] = (10, 10)
        self.options["early_stopping"] = True
        self.options["max_epochs"] = 2
        self.options["segment"] = {"full": "larger_data.tif"}
        self.options["results_dir"] = "../warehouse/results/"
        self.options["dataset_dir"] = "../warehouse/datasets/"

    def run_snakemake(self, cores: int):
        result = subprocess.run(['snakemake', f'-c{cores}'], stdout=subprocess.PIPE, cwd="Ipy")
        print(result.stdout)




if __name__ == '__main__':
    manager = PipeLineManager()
    manager.yaml_constructor("Ipy/config/config.yaml")
    manager.run_full()
    # manager.run_snakemake(cores=4)
