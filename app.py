import os
import streamlit as st
import time
import queue
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Global variables
gesture_controller = None
is_running = False  # Track if tracking is currently running
status_queue = queue.Queue()  # Queue for GUI status updates

# Import GestureController lazily to avoid issues on load
def load_gesture_controller():
    global gesture_controller
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from virtual_mouse import GestureController  # Lazy loading
    gesture_controller = GestureController()
    return "Initialization complete. Starting tracking..."

# Streamlit GUI setup
st.title("AI Virtual Mouse")
st.write("Welcome to Virtual Mouse - Hand Gesture Tracking")

# Status display
status_text = st.empty()

# Define start and stop buttons
if st.button("Track Mouse"):
    # Only start if not already running
    if not is_running:
        status_text.write("Initializing... Please wait.")
        try:
            # Initialize and start the gesture controller
            load_status = load_gesture_controller()
            gesture_controller.start()  # Begin tracking gestures
            is_running = True
            status_queue.put("Tracking started!")
            status_text.write(load_status)
        except Exception as e:
            status_queue.put(f"Initialization failed: {e}")
            st.error(f"Error initializing virtual mouse: {e}")
    else:
        st.warning("Tracking is already running.")

# Stop button for stopping gesture tracking
if st.button("Stop Tracking"):
    if is_running:
        gesture_controller.stop()  # Stop the GestureController
        is_running = False
        status_queue.put("Tracking stopped.")
        status_text.write("Tracking stopped.")
    else:
        st.warning("Tracking is not running.")

# Display status messages
while not status_queue.empty():
    message = status_queue.get_nowait()
    status_text.write(message)
