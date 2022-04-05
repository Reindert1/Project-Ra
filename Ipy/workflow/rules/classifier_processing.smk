def get_input_image_location(wildcards):
    return config["datadir"] + config["classifiers"][wildcards.image]

rule scale_colors:
    input:
        image_location = get_input_image_location
        #config["train_data"]
    output:
        #config["dataset_dir"] + "data_subset/windows.npy"
        config["dataset_dir"] + "classifiers/{image}_color_cleaned.tif"
    script:
        "../scripts/classifier_cleaner.py"

rule combine_classifiers:
    input:
        expand(config["dataset_dir"] + "classifiers/{image}_color_cleaned.tif", image=config["classifiers"])
        #config["train_data"]
    output:
        #config["dataset_dir"] + "data_subset/windows.npy"
        # temp_memmap = config["dataset_dir"] + "classifiers/temp_classifier.npy",
        # temp_memmap_npy = config["dataset_dir"] + "classifiers/temp_classifier_npy.npy",
        full=config["dataset_dir"] + "classifiers/full_classifier.npy"
    params:
        classifiers=config["classifiers"],
        temp_memmap = config["dataset_dir"] + "classifiers/temp_classifier.npy",
        temp_memmap_npy = config["dataset_dir"] + "classifiers/temp_classifier_npy.npy"
    script:
        "../scripts/classifier_encoder.py"