def get_original_image_location(wildcards):
    return config["datadir"] + ALL_DATA[wildcards.classifier] #config["datadir"] + config["classifiers"][wildcards.classifier]

rule classifier_to_tif:
    input:
        model=config["dataset_dir"] + "models/{model_name}.sav",
        dataset=config["dataset_dir"] + "dataset/total_features_{classifier}.npy",
    output:
        config["results_dir"] + "images/classified/{classifier}_{model_name}.tif"
    params:
        original_image_location=get_original_image_location
    message:
        "Building segmented tif image from trained {wildcards.model_name} classifier for {wildcards.classifier}"
    log:
        notebook=config["results_dir"] + "logs/classifier_to_tif/{classifier}_{model_name}.log"
    script:
        "../scripts/classifier_to_tiff.py"

rule image_overlayer:
    input:
        background=get_original_image_location,
        overlay=config["results_dir"] + "images/classified/{classifier}_{model_name}.tif",
    output:
        config["results_dir"] + "images/overlayed/{classifier}_{model_name}_overlay.tif"
    message:
        "Overlaying segmented {wildcards.classifier} from {wildcards.model_name} model with the original image"
    log:
        notebook=config["results_dir"] + "logs/image_overlayer/{classifier}_{model_name}.log"
    script:
        "../scripts/image_overlayer.py"