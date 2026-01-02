import os
import numpy as np
import pandas as pd
import sys
import pickle  # Added for saving/loading
from tkinter import Tk, Label, Button, filedialog, messagebox, Canvas, Text, Scrollbar
from PIL import Image, ImageTk
from minisom import MiniSom
from collections import Counter # Added for Label Map logic

# -------- Global Variables & Constants --------
uploaded_image_vector = None
imgtk = None
som = None
train_images = None
train_labels = None
canvas = None
nutrients_text_area = None
fruit_nutrients = {}
neuron_label_map = {} # This will now store our "Map"

# Constants
# UPDATED: Changed 'Jack Fruit' to 'Jackfruit' to match your CSV file exactly
FRUIT_CLASSES = ['Apple', 'Banana', 'Orange', 'Dragon Fruit', 'Lichi', 'Mango', 'Pineapple', 'Watermelon', 'Guava', 'Jackfruit']
IMAGE_SIZE = (64, 64)
IMAGE_FOLDER = "./fruits"
CSV_FILE = "fruit_nutrients.csv"
MODEL_FILE = "som_model.pkl" # New constant for the saved model file

# Classification Threshold
UNIDENTIFIED_THRESHOLD = 60 

# -------- Data Loading Functions --------

def load_nutrients_from_csv(csv_path):
    """Loads fruit nutritional data from a CSV file into a dictionary."""
    global fruit_nutrients
    try:
        df = pd.read_csv(csv_path)
        # Ensure fruit names are lowercase for consistent lookup
        df['Fruit'] = df['Fruit'].str.lower()
        df.set_index('Fruit', inplace=True)
        fruit_nutrients = df.to_dict('index')
        print(f"Loaded nutrients for {len(fruit_nutrients)} fruits.")
        return True
    except FileNotFoundError:
        print(f"CRITICAL ERROR: CSV file '{csv_path}' not found.")
        return False
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        return False

def load_images_from_folder(folder, size=IMAGE_SIZE):
    """Loads and processes training images."""
    images = []
    labels = []
    
    if not os.path.exists(folder):
        print(f"Error: The folder '{folder}' does not exist.")
        return np.array([]), np.array([])

    for label in FRUIT_CLASSES:
        class_folder = os.path.join(folder, label)
        # Try finding folder with lowercase if Capitalized not found (robustness)
        if not os.path.exists(class_folder):
            class_folder = os.path.join(folder, label.lower())
        
        if not os.path.exists(class_folder):
            print(f"Warning: Class folder for '{label}' not found. Skipping.")
            continue
            
        for filename in os.listdir(class_folder):
            path = os.path.join(class_folder, filename)
            try:
                img = Image.open(path).convert('RGB')
                img = img.resize(size)
                images.append(np.array(img).flatten() / 255.0)
                labels.append(label)
            except Exception:
                pass
                
    return np.array(images), np.array(labels)

# -------- Model Handling (Train/Save/Load) --------

def train_som_model(images, labels):
    """Trains the SOM and generates the Label Map."""
    global neuron_label_map
    print("Training SOM model... please wait.")
    
    som_model = MiniSom(10, 10, images.shape[1], sigma=3.0, learning_rate=0.5)
    som_model.random_weights_init(images)
    som_model.train_random(images, 5000)
    
    print("Training complete. Generating Label Map...")
    # Generate the map: Which neuron represents which fruit?
    raw_map = som_model.labels_map(images, labels)
    neuron_label_map = {}
    
    for position, label_list in raw_map.items():
        # Resolve conflicts: If a neuron hit 'apple' 5 times and 'banana' 1 time, it is an 'apple' neuron.
        most_common_label = Counter(label_list).most_common(1)[0][0]
        neuron_label_map[position] = most_common_label
        
    print("Label Map Generated!")
    return som_model

def save_model_data(som_model, label_map):
    """Saves the trained model and label map to a file using pickle."""
    data = {
        "model": som_model,
        "map": label_map
    }
    try:
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(data, f)
        print(f"Model saved successfully to {MODEL_FILE}")
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model_data():
    """Loads the model and label map from file."""
    try:
        with open(MODEL_FILE, 'rb') as f:
            data = pickle.load(f)
        print(f"Model loaded from {MODEL_FILE}")
        return data["model"], data["map"]
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

# -------- GUI Functions --------

def display_nutrients(label):
    """Updates the GUI text area with nutritional info."""
    global nutrients_text_area
    nutrients_text_area.delete('1.0', 'end')
    
    # Clean the label to match CSV keys
    fruit_name = label.lower().strip()
    
    if "unidentified" in fruit_name:
        nutrients_text_area.insert('1.0', "🚫 Could not identify a fruit match.\nDistance was too far from known categories.")
        return
        
    if fruit_name in fruit_nutrients:
        data = fruit_nutrients[fruit_name]
        info_string = f"🍎 Nutritional Facts for: {fruit_name.upper()} 🍏\n\n"
        for key, value in data.items():
            info_string += f"{key}: {value}\n"
        nutrients_text_area.insert('1.0', info_string)
    else:
        nutrients_text_area.insert('1.0', f"Nutritional data not found for '{fruit_name.upper()}' in the CSV.")

def browse_image():
    """Opens a file dialog, loads the image, and displays it."""
    global uploaded_image_vector, imgtk
    file_path = filedialog.askopenfilename(title="Select a fruit image")
    if file_path:
        try:
            img = Image.open(file_path).convert("RGB").resize(IMAGE_SIZE)
            uploaded_image_vector = np.array(img).flatten() / 255.0
            
            img_display = Image.open(file_path).resize((180, 180))
            imgtk = ImageTk.PhotoImage(img_display)
            canvas.create_image(0, 0, anchor='nw', image=imgtk)
        except Exception as e:
            messagebox.showerror("Image Error", f"Could not load image: {e}")

def classify_image():
    """OPTIMIZED: Uses the pre-calculated Label Map for instant classification."""
    if uploaded_image_vector is None:
        messagebox.showwarning("Warning", "Please upload an image first.")
        return

    if som is None:
        messagebox.showerror("Error", "Model is not ready.")
        return

    try:
        # 1. Find the winning neuron
        winner = som.winner(uploaded_image_vector)
        
        # 2. Check Distance (Outlier Detection)
        winning_weights = som.get_weights()[winner]
        quantization_error = np.linalg.norm(uploaded_image_vector - winning_weights)
        
        if quantization_error > UNIDENTIFIED_THRESHOLD:
            messagebox.showwarning("Result", f"⚠️ Unidentified Object (Distance: {quantization_error:.2f})")
            display_nutrients("Unidentified Object")
            return

        # 3. Fast Classification using Map
        if winner in neuron_label_map:
            predicted_label = neuron_label_map[winner]
        else:
            # If the neuron was empty during training, find the nearest labeled neuron
            min_dist = float('inf')
            closest_label = "Unknown"
            
            for position, label in neuron_label_map.items():
                dist = np.linalg.norm(np.array(winner) - np.array(position))
                if dist < min_dist:
                    min_dist = dist
                    closest_label = label
            predicted_label = closest_label

        # 4. Display Result
        messagebox.showinfo("Classification Result", f"✅ The image is classified as: {predicted_label.upper()}")
        display_nutrients(predicted_label)
        
    except Exception as e:
        print(e)
        messagebox.showerror("Classification Error", f"Could not classify the image: {e}")

def start_gui():
    """Sets up and runs the main Tkinter GUI window."""
    global canvas, nutrients_text_area
    root = Tk()
    root.title("SOM Fruit Classifier & Nutrients")
    root.geometry("380x600")

    Label(root, text="Upload a fruit image to classify", font=("Arial", 12)).pack(pady=10)

    canvas = Canvas(root, width=180, height=180, bg="#e0e0e0")
    canvas.pack()

    btn_frame = Label(root)
    btn_frame.pack(pady=10)
    
    Button(btn_frame, text="Browse Image", command=browse_image).pack(side="left", padx=5)
    Button(btn_frame, text="Classify Fruit", command=classify_image).pack(side="left", padx=5)
    
    Label(root, text="--- Fruit Nutritional Information ---", font=("Arial", 10, 'bold')).pack(pady=5)
    
    text_frame = Label(root)
    text_frame.pack(padx=10, pady=5)
    
    scrollbar = Scrollbar(text_frame)
    scrollbar.pack(side="right", fill="y")

    nutrients_text_area = Text(text_frame, height=8, width=40, wrap='word', 
                               yscrollcommand=scrollbar.set, font=("Courier", 10))
    nutrients_text_area.pack(side="left", fill="both")
    scrollbar.config(command=nutrients_text_area.yview)

    root.mainloop()

# -------- Main Entry --------
if __name__ == "__main__":
    
    # 1. Load nutritional data
    if not load_nutrients_from_csv(CSV_FILE):
        pass 

    # 2. Check for existing model
    if os.path.exists(MODEL_FILE):
        print("Found saved model. Loading...")
        som, neuron_label_map = load_model_data()
    else:
        print("No saved model found. Initiating training...")
        
        # Load images
        train_images, train_labels = load_images_from_folder(IMAGE_FOLDER)
        
        if len(train_images) == 0:
            print("CRITICAL ERROR: No training images found.")
            print(f"Please ensure '{IMAGE_FOLDER}' contains subfolders for each fruit.")
        else:
            # Train and Save
            som = train_som_model(train_images, train_labels)
            save_model_data(som, neuron_label_map)
        
    # 3. Start the application
    start_gui()