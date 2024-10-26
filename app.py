import os
import streamlit as st
import time
import threading
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Global variables
gesture_controller = None
is_running = False  # Track if tracking is currently running

# Import GestureController lazily to avoid issues on load
def load_gesture_controller():
    global gesture_controller
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from virtual_mouse import GestureController  # Lazy loading
    gesture_controller = GestureController()
    return "Initialization complete. Starting tracking..."

# Function to run gesture tracking in a separate thread
def start_tracking():
    global is_running
    try:
        load_gesture_controller()  # Load the gesture controller
        gesture_controller.start()  # Begin tracking gestures
        is_running = True
        st.session_state.status_text = "Tracking started!"
    except Exception as e:
        st.session_state.status_text = f"Error initializing virtual mouse: {e}"

# Streamlit GUI setup
st.title("AI Virtual Mouse")
st.write("Welcome to Virtual Mouse - Hand Gesture Tracking")

# Initialize session state for status text
if 'status_text' not in st.session_state:
    st.session_state.status_text = ""

# Display status messages
st.write(st.session_state.status_text)

# Define start and stop buttons
if st.button("Track Mouse"):
    if not is_running:
        st.session_state.status_text = "Initializing... Please wait."
        status_thread = threading.Thread(target=start_tracking)
        status_thread.start()  # Start tracking in a separate thread
    else:
        st.warning("Tracking is already running.")

# Stop button for stopping gesture tracking
if st.button("Stop Tracking"):
    if is_running:
        gesture_controller.stop()  # Stop the GestureController
        is_running = False
        st.session_state.status_text = "Tracking stopped."
    else:
        st.warning("Tracking is not running.")
