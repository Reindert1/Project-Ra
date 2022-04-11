def get_input_image_location(wildcards):
    return config["datadir"] + ALL_DATA[wildcards.image]

rule build_gaussian_pyramid:
    input:
        image_location = get_input_image_location
    output:
        config["dataset_dir"] + "data_subset/{image}_gaussian.npy"
    params:
        gaussian_layers=config["gaussian_layers"]
    threads:
        1
    message:
        "Builing gaussian pyramids dataset for {wildcards.image}"
    log:
        config["results_dir"] + "logs/build_gaussian_pyramid/{image}.log"
    benchmark:
        config["results_dir"] + "benchmarks/build_gaussian_pyramid/{image}.benchmark.txt"
    script:
        "../scripts/dataset_building/gaussian_builder.py"

rule roll_windows:
    input:
        image_location = get_input_image_location
    output:
        config["dataset_dir"] + "data_subset/{image}_windows.npy"
    params:
        window_size=config["window_size"]
    threads:
        1
    message:
        "Building neigboring pixel dataset for {wildcards.image}"
    log:
        config["results_dir"] + "logs/roll_windows/{image}.log"
    benchmark:
        config["results_dir"] + "benchmarks/roll_windows/{image}.benchmark.txt"
    script:
        "../scripts/dataset_building/window_roller.py"
