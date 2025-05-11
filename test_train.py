import unsloth
from unsloth import FastVisionModel
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import pandas as pd
import json
import shutil
from PIL import Image
import torch
from transformers import TextStreamer
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import threading
import os
import zipfile
from ultralytics import YOLO
import cv2
import time

app = Flask(__name__)
CORS(app)

# Set folders
TEST_UPLOAD_FOLDER = 'test_images'
MODEL_BASE_FOLDER = "Model"
DATASET_FOLDER = 'dataset'
YOLO_DATASET_FOLDER = 'yolo_dataset'

# Ensure base folders exist
os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(YOLO_DATASET_FOLDER, exist_ok=True)
os.makedirs(TEST_UPLOAD_FOLDER, exist_ok=True)

# Global variable to store model name
current_model_name = None

@app.route('/')
def home():
    return render_template('index.html')  # It will load your index.html


# @app.route('/get_models')
# def get_models():
#     try:
#         models = [
#             d for d in os.listdir(MODEL_BASE_FOLDER)
#             if os.path.isdir(os.path.join(MODEL_BASE_FOLDER, d))
#         ]
#         return jsonify(models=models)
#     except Exception as e:
#         return jsonify(error=str(e)), 500
@app.route('/get_models')
def get_models():
    try:
        models = []
        for d in os.listdir(MODEL_BASE_FOLDER):
            full_path = os.path.join(MODEL_BASE_FOLDER, d)
            if os.path.isdir(full_path):
                # If the model name ends with "_yolo", remove it
                if d.endswith('_yolo'):
                    models.append(d[:-5])  # Remove last 5 characters
                else:
                    models.append(d)
        return jsonify(models=models)
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/results_image/<model_name>')
def serve_results_image(model_name):
    model_folder = os.path.join(MODEL_BASE_FOLDER, model_name)
    image_path = os.path.join(model_folder, 'results.png')
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/png')
    else:
        return jsonify({'error': 'Result image not found'}), 404

@app.route('/train', methods=['POST'])
def train_model():
    # global current_model_name
    model = request.form.get("model", "unsloth/Llama-3.2-11B-Vision-Instruct")
    print(f"Selected model: {model}")
    # current_model_name = model  # Store the model name globally
    

    if "yolo" in model.lower():
        # Get inputs from frontend ---------------------------------------------------------------
        # Get the uploaded ZIP file
        zip_file = request.files.get('yolo_dataset')
        if not zip_file or not zip_file.filename.endswith('.zip'):
            return jsonify({'error': 'Please upload a valid ZIP file'}), 400

        # Get custom model name
        user_model_name = secure_filename(request.form.get('yolo_output_dir'))
        if not user_model_name:
            return jsonify({'error': 'Custom model name is required'}), 400
        
        try:
            epochs = int(request.form.get('epochs', 50))
        except (ValueError, TypeError):
            return jsonify({'error': 'Epochs must be a valid integer'}), 400

        try:
            imgsz = int(request.form.get('imgz', 640))
        except (ValueError, TypeError):
            return jsonify({'error': 'Image size must be a valid integer'}), 400
        
        # ------------------------------------------------------------------------------------------------

        # Add internal suffix to model name for saving
        # custom_model_name = f"{user_model_name}_yolo"
        # Add internal suffix to model name for saving
        custom_model_name = f"{user_model_name}_yolo"
        model_output_path = os.path.join("/root/Desktop/AI_ML_Applications/Local_vision/flaskapp/Model", custom_model_name)

        # Delete existing model folder if it already exists
        if os.path.exists(model_output_path):
            shutil.rmtree(model_output_path)
            print(f"⚠️ Existing model folder '{model_output_path}' deleted to allow overwrite.")


        # Save the uploaded ZIP temporarily
        temp_zip_path = os.path.join('/tmp', zip_file.filename)
        zip_file.save(temp_zip_path)

        def extract_zip_strip_top_folder(zip_path, destination_folder):
            os.makedirs(destination_folder, exist_ok=True)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                members = zip_ref.namelist()

                # Detect the top-level folder (e.g., 'Shop_boards_with_down.v20i.yolov11/')
                top_folder = os.path.commonprefix(members).rstrip('/')

                for member in members:
                    # Remove top-level folder from path
                    member_path = member[len(top_folder)+1:] if member.startswith(top_folder + '/') else member

                    # Skip empty paths
                    if not member_path.strip():
                        continue

                    # Create full destination path
                    target_path = os.path.join(destination_folder, member_path)

                    # Create directories or extract files
                    if member.endswith('/'):
                        os.makedirs(target_path, exist_ok=True)
                    else:
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        with open(target_path, 'wb') as f:
                            f.write(zip_ref.read(member))

            print(f"Contents from '{zip_path}' extracted to '{destination_folder}' without top-level folder.")
        
        extract_zip_strip_top_folder(temp_zip_path, YOLO_DATASET_FOLDER)
        # delete the temp zip file
        os.remove(temp_zip_path)

        # model training
        save_project_path = "/root/Desktop/AI_ML_Applications/Local_vision/flaskapp/Model"
        data_yaml_path = "/root/Desktop/AI_ML_Applications/Local_vision/flaskapp/yolo_dataset/data.yaml"

        def train_custom_yolo_model(base_model_name: str, custom_model_name: str, epochs, imgsz):
            
            # Load the specified model
            model = YOLO(base_model_name)

            # Train the model and save results in specified folder
            results = model.train(
                data=data_yaml_path,
                epochs=epochs,
                imgsz=imgsz,
                project=save_project_path,
                name=custom_model_name
            )

            print(f"\n✅ Training complete! Results saved at: {results.save_dir}")
            return results
        
        # time.sleep(10)
        train_custom_yolo_model(model, custom_model_name, epochs, imgsz)

    else:
        instruction = request.form.get('instruction')
        # output_dir = request.form.get('output_dir')

        output_dir_name = secure_filename(request.form.get('output_dir'))
        output_dir = os.path.join("Model", output_dir_name)

        if not instruction or not output_dir:
            return jsonify({'error': 'Instruction and Output Directory are required'}), 400
        
        # Create dataset/images and dataset/labels folders
        dataset_images_folder = os.path.join(DATASET_FOLDER, 'images')
        dataset_labels_folder = os.path.join(DATASET_FOLDER, 'labels')

        os.makedirs(dataset_images_folder, exist_ok=True)
        os.makedirs(dataset_labels_folder, exist_ok=True)

        # Handle uploaded files
        images = request.files.getlist('images')
        labels = request.files.getlist('labels')
        classes_file = request.files.get('classes')
        notes_file = request.files.get('notes')

        # Save images
        for img in images:
            filename = secure_filename(img.filename)
            img.save(os.path.join(dataset_images_folder, filename))

        # Save labels
        for lbl in labels:
            filename = secure_filename(lbl.filename)
            lbl.save(os.path.join(dataset_labels_folder, filename))

        # Save classes.txt
        if classes_file:
            classes_path = os.path.join(DATASET_FOLDER, 'classes.txt')
            classes_file.save(classes_path)

        # Save notes.json if uploaded
        if notes_file:
            notes_path = os.path.join(DATASET_FOLDER, 'notes.json')
            notes_file.save(notes_path)

        # Create the output directory
        # final_output_dir = secure_filename(output_dir)
        # os.makedirs(final_output_dir, exist_ok=True)

        os.makedirs(MODEL_BASE_FOLDER, exist_ok=True)

        os.makedirs(output_dir, exist_ok=True)
        final_output_dir = output_dir  # Directly use the right path

        # === Step 1: Convert YOLO annotations to image-caption pairs ===

        yolo_labels_dir = "dataset/labels"
        image_dir = "dataset/images"
        classes_path = "dataset/classes.txt"
        notes_path = "dataset/notes.json"
        output_excel = "output_excel.xlsx"


        with open(classes_path, "r") as f:
                classes = [line.strip() for line in f.readlines()]
            
            
        # Load notes.json if available
        notes = {}
        if notes_path and os.path.exists(notes_path):
            with open(notes_path, "r") as f:
                notes = json.load(f)
        

        data = []
        for label_file in os.listdir(yolo_labels_dir):
            if not label_file.endswith(".txt"):
                continue

            base_name = label_file.replace(".txt", "")
            possible_extensions = [".jpg", ".jpeg", ".png"]
            image_file = None
            image_path = None

            for ext in possible_extensions:
                candidate_path = os.path.join(image_dir, base_name + ext)
                if os.path.exists(candidate_path):
                    image_file = base_name + ext
                    image_path = candidate_path
                    break

            if not image_path:
                continue  # Skip if image doesn't exist

            label_path = os.path.join(yolo_labels_dir, label_file)
            labels = set()

            with open(label_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    class_id = int(line.split()[0])
                    if 0 <= class_id < len(classes):
                        labels.add(classes[class_id])
                    else:
                        print(f"⚠️ Invalid class_id {class_id} found in {label_path}, skipping.")


            if labels:
                caption = "The image contains " + ", ".join(sorted(labels)) + "."
            else:
                caption = "The image contains no notable findings."

            # Append note if available
            note_info = notes.get(image_file, "")
            if note_info:
                caption += f" Additional note: {note_info}"

            data.append({"image_path": image_path, "caption": caption})

            df = pd.DataFrame(data)
            df.to_excel(output_excel, index=False)
            print(f"Saved caption dataset to {output_excel}")

        # Training function
        # model_name="unsloth/Llama-3.2-11B-Vision-Instruct"
        model_name = model
        excel_path="output_excel.xlsx"
        image_column="image_path"
        caption_column="caption"

        try:
            print("📦 Loading model...")

            model, tokenizer = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
            )

            print("📦 Model loaded successfully!")

            model = FastVisionModel.get_peft_model(
                model,
                finetune_vision_layers=False,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=16,
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
            )

            # Load CSV and images
            df = pd.read_excel(excel_path)

            # def load_image(row):
            #     row["image"] = Image.open(row[image_column]).convert("RGB")
            #     row["caption"] = row[caption_column]
            #     return row

            # dataset = Dataset.from_pandas(df)
            # dataset = dataset.map(load_image)

            def load_image(row):
                try:
                    if os.path.exists(row[image_column]):
                        row["image"] = Image.open(row[image_column]).convert("RGB")
                    else:
                        raise FileNotFoundError(f"Image not found: {row[image_column]}")
                except Exception as e:
                    print(f"❌ Failed to load image {row[image_column]}: {e}")
                    row["image"] = None  # Mark bad images
                return row

            dataset = Dataset.from_pandas(df)
            dataset = dataset.map(load_image)

            print("📦 Dataset loaded successfully!")

            # Remove bad images
            dataset = dataset.filter(lambda example: example["image"] is not None)

            def convert_to_conversation(sample):
                return {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": instruction},
                                {"type": "image", "image": sample["image"]}
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": sample["caption"]}
                            ]
                        },
                    ]
                }

            converted_dataset = [convert_to_conversation(sample) for sample in dataset]

            # Switch to training mode
            FastVisionModel.for_training(model)

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                data_collator=UnslothVisionDataCollator(model, tokenizer),
                train_dataset=converted_dataset,
                args=SFTConfig(
                    per_device_train_batch_size=2,
                    gradient_accumulation_steps=4,
                    warmup_steps=5,
                    max_steps=30,
                    learning_rate=2e-4,
                    fp16=not is_bf16_supported(),
                    bf16=is_bf16_supported(),
                    logging_steps=1,
                    optim="adamw_8bit",
                    weight_decay=0.01,
                    lr_scheduler_type="linear",
                    seed=3407,
                    output_dir=output_dir,
                    report_to="none",
                    remove_unused_columns=False,
                    dataset_text_field="",
                    dataset_kwargs={"skip_prepare_dataset": True},
                    dataset_num_proc=4,
                    max_seq_length=2048,
                ),
            )

            trainer.train()

            # Save model & tokenizer
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Model and tokenizer saved to: {output_dir}")


        except Exception as e:
            print(f"❌ Error during training: {str(e)}")

    
    return jsonify({'message': 'Training completed successfully!'})

@app.route('/infer', methods=['POST'])
def infer_model():
    MODEL_BASE_FOLDER = "/root/Desktop/AI_ML_Applications/Local_vision/flaskapp/Model"  # Change this to your actual path

    # instruction = request.form.get('instruction')
    instruction = "You are an expert in classifying images and identifying objects in images based on training data (training classes). Please classify the image and identify the objects in the image. If no objects are found, please say 'No objects found'."
    output_dir = request.form.get('output_dir')
    image_file = request.files.get('image')

    print("image_file:", image_file)
    print("output_dir:", output_dir)    

    # if not instruction or not output_dir or not image_file:
    #     return jsonify({'error': 'Instruction, Output Directory, and Image are required'}), 400

    save_path = os.path.join(TEST_UPLOAD_FOLDER, secure_filename(image_file.filename))
    image_file.save(save_path)

    print(f"Image saved to: {save_path}")

    # Normal model inference function----------------------------------------------------------------------------------------
    def infer_with_unsloth_vision(model, tokenizer, image_path, instruction):

        # Enable inference mode
        FastVisionModel.for_inference(model)

        # Load the image
        image = Image.open(image_path).convert("RGB")

        # Prepare message
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]}
        ]

        # Prepare input
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

        # Generate output (no streamer)
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            use_cache=True,
            temperature=1.5,
            min_p=0.1
        )

        # Decode the output
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract assistant's part only
        if "assistant" in output_text:
            assistant_response = output_text.split("assistant")[-1].strip()
        else:
            assistant_response = output_text.strip()

        return assistant_response
    
    # Inference function for YOLO model----------------------------------------------------------------------------------------
    def predict_and_save_annotated_image(model_name: str, image_path: str, conf_threshold: float = 0.5):

        # Define base paths
        base_model_path = f"/root/Desktop/AI_ML_Applications/Local_vision/flaskapp/Model/{model_name}/weights/best.pt"
        save_dir = f"/root/Desktop/AI_ML_Applications/Local_vision/flaskapp/static/Model/Predictions/{model_name}"


        # Load model
        model = YOLO(base_model_path)

        # Predict
        results = model.predict(
            source=image_path,
            conf=conf_threshold,
            save=False
        )

        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)

        detected_classes = []  # To store the detected classes

        for i, result in enumerate(results):
            # Extract the classes for this result (assuming results contain class IDs)
            classes = result.names  # Assuming result.names has class names, check your YOLO version

            # Get the class labels for each detection (if any)
            detected_labels = [classes[int(cls_id)] for cls_id in result.boxes.cls]  # `boxes.cls` holds class IDs
            detected_classes.append(detected_labels)

            # Plot the image with annotations
            im_bgr = result.plot()
            save_path = os.path.join(save_dir, f"prediction_{i}.jpg")
            cv2.imwrite(save_path, im_bgr)
            print(f"✅ Annotated image saved at: {save_path}")

            # Output the detected classes for this prediction
            print(f"Detected classes: {detected_labels}")
        
        # Construct the public path to the prediction image
        image_filename = f"prediction_0.jpg"
        image_rel_path = f"Model/Predictions/{model_name}/{image_filename}"
        image_url = f"/static/{image_rel_path}"

        # return detected_classes  # Return the detected classes
        result = {
            "prediction": detected_classes,
            "image": image_url
        }
        return result

    # ---------------------------------------------------------------------------------------------------------------------------------
    
    models_with_yolo = [
        model for model in os.listdir(MODEL_BASE_FOLDER)
        if os.path.isdir(os.path.join(MODEL_BASE_FOLDER, model)) and model.endswith('_yolo')
    ]

    # # Try to match user-provided name (exact or partial match)
    matched_model = next(
        (model for model in models_with_yolo if output_dir == model or output_dir in model),
        None
    )

    print(f"Matched model: {matched_model}")

    if matched_model:
        print(f"Detected YOLO model: {matched_model}")
        result = predict_and_save_annotated_image(matched_model, save_path)
        print(f"Result: {result}")
    else:
        print(f"Not Detected YOLO model")

        # model_path = secure_filename(output_dir)
        model_path = os.path.join("Model", secure_filename(output_dir))

        if not os.path.exists(model_path):
            return jsonify({'error': f'Model output directory {model_path} does not exist.'}), 400
        
        # Load your fine-tuned model
        model, tokenizer = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        )

        # Step 2: Move manually
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        result = infer_with_unsloth_vision(
                model=model,
                tokenizer=tokenizer,
                image_path=image_file,
                instruction=instruction
                )
        
        print(f"Result: {result}")
        # cleaned_result = result.split('<|eot_id|>')[0].strip()
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5021, use_reloader=False)

# use_reloader=False
# kill -9 475898
# nvidia-smi