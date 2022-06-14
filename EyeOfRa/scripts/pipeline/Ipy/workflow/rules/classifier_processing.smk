def get_input_image_location(wildcards):
    return config["datadir"] + config["classifiers"][wildcards.image]

rule scale_colors:
    input:
        image_location = get_input_image_location
    output:
        config["dataset_dir"] + "classifiers/{image}_color_cleaned.tif"
    threads:
        1
    message:
        "Scaling colors for {wildcards.image}"
    log:
        config["results_dir"] + "logs/scale_colors/{image}.log"
    benchmark:
        config["results_dir"] + "benchmarks/scale_colors/{image}.benchmark.txt"
    script:
        "../scripts/dataset_building/classifier_cleaner.py"

rule combine_classifiers:
    input:
        expand(config["dataset_dir"] + "classifiers/{image}_color_cleaned.tif", image=config["classifiers"])
    output:
        full=config["dataset_dir"] + "classifiers/full_classifier.npy"
    params:
        classifiers=config["classifiers"],
        temp_memmap = config["dataset_dir"] + "classifiers/temp_classifier.npy",
        temp_memmap_npy = config["dataset_dir"] + "classifiers/temp_classifier_npy.npy"
    threads:
        1
    message:
        "Building full classifier"
    log:
        config["results_dir"] + "logs/combine_classifiers/full_classifier.log"
    benchmark:
        config["results_dir"] + "benchmarks/combine_classifiers/full_classifier.benchmark.txt"
    script:
        "../scripts/dataset_building/classifier_encoder.py"