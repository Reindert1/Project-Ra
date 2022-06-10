def get_original_image_location_nn(wildcards):
    return config["datadir"] + ALL_DATA[wildcards.classifier]

rule predict:
    input:
        model_location = config["results_dir"] + "model/trained_model.h5",
        data_location = config["dataset_dir"] + "dataset/total_features_{classifier}.npy",
        original_location = get_original_image_location

    output:
        save_location = config["results_dir"] + "images/classified/NN/{classifier}.png"

    script:
        "../scripts/deep_learning/predict_and_render_image.py"
