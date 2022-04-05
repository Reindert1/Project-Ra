def get_original_image_location(wildcards):
    return config["datadir"] + ALL_DATA[wildcards.classifier] #config["datadir"] + config["classifiers"][wildcards.classifier]

rule classifier_to_tif:
    input:
        model=config["dataset_dir"] + "models/{model_name}.sav",
        dataset=config["dataset_dir"] + "dataset/total_features_{classifier}.npy",
    output:
        config["dataset_dir"] + "images/{classifier}_{model_name}.tif"
    params:
        original_image_location=get_original_image_location
    script:
        "../scripts/classifier_to_tiff.py"