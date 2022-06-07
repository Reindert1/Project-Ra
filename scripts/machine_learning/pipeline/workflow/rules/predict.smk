rule predict:
    input:
        model_location = config["trained_model"],
        data_location = config["input_data"],
        original_location = config["original_image_location"]

    output:
        save_location = config["classified_image"]

    script:
        "../scripts/predict_and_render_image.py"
