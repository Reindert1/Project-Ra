rule predict:
    input:
        model_location = config["model_export"] + "trained_model.h5",
        data_location = config["input_data"],
        original_location = config["original_image_location"]

    output:
        save_location = config["classified_image"]

    script:
        "../scripts/predict_and_render_image.py"
