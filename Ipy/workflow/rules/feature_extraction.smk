def get_input_image_location(wildcards):
    return config["datadir"] + ALL_DATA[wildcards.image]

rule build_gaussian_pyramid:
    input:
        #config["train_data"]
        #expand("{images}", images=config["segment"])
        #image_location = config["data_dir"] + {image}
        image_location = get_input_image_location
    output:
        #config["dataset_dir"] + "data_subset/gaussian.npy"
        config["dataset_dir"] + "data_subset/{image}_gaussian.npy"
    params:
        gaussian_layers=config["gaussian_layers"]
    script:
        "../scripts/gaussian_builder.py"

rule roll_windows:
    input:
        image_location = get_input_image_location
        #config["train_data"]
    output:
        #config["dataset_dir"] + "data_subset/windows.npy"
        config["dataset_dir"] + "data_subset/{image}_windows.npy"
    params:
        window_size=config["window_size"]
    script:
        "../scripts/window_roller.py"
