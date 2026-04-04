# 🤟 ASL Learning Game
**Real-time American Sign Language recognition with a gamified experience.**

This application uses **Computer Vision** and **Deep Learning** to create an interactive way to learn the ASL alphabet. Built with Streamlit, OpenCV, and Google’s Mediapipe, it provides instant feedback on your hand signs.

---

## ✨ Features
* **Real-time Detection:** High-speed hand tracking and classification.
* **Gamified UI:** Track your current **Score**, **Streak**, and **Best Streak**.
* **Hold to Confirm:** Includes a "Hold Progress" mechanic to ensure accuracy before awarding points.
* **Visual Hints:** Displays reference images for the target letter to help you learn.
* **Dark Mode Aesthetic:** A custom-designed CSS interface with modern typography.

## 🛠️ Tech Stack
* **Python 3.9+**
* **Streamlit:** Web-based game interface.
* **OpenCV:** Video stream processing.
* **cvzone:** Hand Tracking and Keras model classification.
* **TensorFlow/Keras:** Deep learning model backend.

## 🚀 Getting Started

### 1. Installation
To avoid version conflicts (especially with NumPy 2.0 and Keras 3.0), use the following versions:
```bash
pip install "numpy<2"
pip install "tensorflow<2.16"
pip install opencv-python cvzone mediapipe streamlit
```
2. File Structure
Organize your project folder as follows:

```
├── app.py                # The main Python code
├── keras_model.h5        # Your trained Keras model
├── labels.txt            # List of labels (A-Z)
└── signs-imgs/           # Folder containing hint images (A.png, B.png, etc.)
```

3. Run the Game
Navigate to your project folder and run:

```Bash
streamlit run streamlit-asl-alphabet-detection.py
```
🎮 How to Play

* Start the Camera: Click the ▶ Start button in the header.
* Match the Target: Look at the "Target Letter" card and show that sign to your webcam.
* Hold Steady: Once the correct sign is detected, a green progress bar will fill up. Hold it until it hits 100%!
* Level Up: Your score and streak will increase, and a new letter will be assigned automatically.
* Stuck? Check the "Reference" image at the bottom right for a visual hint.

⚙️ Configuration

* Use the Sidebar inside the application to:
* Update the Model Path and Labels Path.
* Set the Signs Images Folder path.
* Adjust the Hold Frames slider (how long you must hold a sign for it to count).

📝 License
This project was created for educational purposes. Feel free to fork, improve, and share!
