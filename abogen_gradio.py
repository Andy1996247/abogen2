import sys
import os
import threading
import time
import shutil
from unittest.mock import MagicMock, Mock

# --- Mock PyQt6 Modules BEFORE importing abogen ---
# This is crucial because abogen imports PyQt6 at the top level.

# Mock PyQt6.QtCore
class MockSignal:
    def __init__(self, *args):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def emit(self, *args):
        for callback in self.callbacks:
            try:
                callback(*args)
            except Exception as e:
                print(f"Error in mock signal callback: {e}")

class MockQThread(threading.Thread):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent

    def start(self, priority=None):
        super().start()

    def quit(self):
        pass

    def wait(self):
        self.join()

class MockQt:
    class WindowType:
        WindowCloseButtonHint = 1
        WindowContextHelpButtonHint = 2
    
    class Key:
        Key_Escape = 1
    
    class CheckState:
        Checked = 2
        Unchecked = 0

class MockQLibraryInfo:
    class LibraryPath:
        PluginsPath = 1
    
    @staticmethod
    def path(p):
        return "/tmp"

# Create the mock modules
mock_qt_core = MagicMock()
mock_qt_core.QThread = MockQThread
mock_qt_core.pyqtSignal = MockSignal
mock_qt_core.Qt = MockQt
mock_qt_core.QTimer = MagicMock()
mock_qt_core.QLibraryInfo = MockQLibraryInfo

mock_qt_widgets = MagicMock()
mock_qt_widgets.QApplication = MagicMock()
mock_qt_widgets.QDialog = MagicMock()
mock_qt_widgets.QLabel = MagicMock()
mock_qt_widgets.QVBoxLayout = MagicMock()
mock_qt_widgets.QCheckBox = MagicMock()
mock_qt_widgets.QDialogButtonBox = MagicMock()

mock_qt_gui = MagicMock()
mock_qt_gui.QIcon = MagicMock()
mock_qt_gui.QtMsgType = MagicMock()
mock_qt_gui.qInstallMessageHandler = MagicMock()

# Inject mocks into sys.modules
sys.modules["PyQt6"] = MagicMock()
sys.modules["PyQt6.QtCore"] = mock_qt_core
sys.modules["PyQt6.QtWidgets"] = mock_qt_widgets
sys.modules["PyQt6.QtGui"] = mock_qt_gui

# --- Now we can import abogen modules ---
# Add current directory to sys.path to ensure imports work
sys.path.append(os.getcwd())

from abogen.conversion import ConversionThread, load_numpy_kpipeline
from abogen.constants import (
    LANGUAGE_DESCRIPTIONS,
    VOICES_INTERNAL,
    SUPPORTED_SOUND_FORMATS,
    SUPPORTED_SUBTITLE_FORMATS
)
import gradio as gr

# --- Global Variables ---
conversion_thread = None
logs = []

# --- Helper Functions ---

def get_language_choices():
    return [f"{code} - {desc}" for code, desc in LANGUAGE_DESCRIPTIONS.items()]

def get_voice_choices():
    return VOICES_INTERNAL

def log_callback(msg):
    global logs
    # Handle tuple messages (msg, color)
    if isinstance(msg, tuple):
        msg = msg[0]
    logs.append(str(msg))
    # Keep logs manageable
    if len(logs) > 1000:
        logs = logs[-1000:]

def progress_callback(percent, etr):
    # This will be used to update Gradio progress
    pass

def run_conversion(
    text_input,
    file_input,
    language_str,
    voice,
    speed,
    output_format,
    subtitle_mode,
    subtitle_format,
    split_pattern,
    progress=gr.Progress()
):
    global conversion_thread, logs
    logs = [] # Clear logs
    
    # Parse language code
    lang_code = language_str.split(" - ")[0]
    
    # Determine input
    file_name = ""
    is_direct_text = False
    
    if file_input is not None:
        file_name = file_input.name
    elif text_input.strip():
        # Create a temp file for text input or pass directly?
        # ConversionThread handles direct text if is_direct_text is True
        # But ConversionThread logic for is_direct_text seems to expect file_name to be the text
        file_name = text_input
        is_direct_text = True
    else:
        return "Error: Please provide text or a file.", None

    # Load pipeline (this might take a moment)
    logs.append("Loading Kokoro pipeline...")
    try:
        np_module, kpipeline_class = load_numpy_kpipeline()
    except Exception as e:
        return f"Error loading pipeline: {e}", None

    # Create output directory
    output_folder = os.path.join(os.getcwd(), "output")
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize ConversionThread
    # Note: We need to handle the arguments carefully matching __init__
    
    # Mocking some GUI-specific attributes that might be accessed
    # ConversionThread expects to emit signals, which we mocked.
    
    try:
        thread = ConversionThread(
            file_name=file_name,
            lang_code=lang_code,
            speed=float(speed),
            voice=voice,
            save_option="Choose output folder", # Mocked
            output_folder=output_folder,
            subtitle_mode=subtitle_mode,
            output_format=output_format,
            np_module=np_module,
            kpipeline_class=kpipeline_class,
            start_time=time.time(),
            total_char_count=len(text_input) if is_direct_text else 1000, # Estimate or calc
            use_gpu=True, # Try to use GPU
            from_queue=False
        )
        
        # Set specific attributes
        thread.is_direct_text = is_direct_text
        thread.subtitle_format = subtitle_format
        thread.merge_chapters_at_end = True # Default to merge
        thread.save_chapters_separately = False
        thread.replace_single_newlines = False
        
        # Connect signals
        # We need a way to capture the output path from conversion_finished
        result_container = {"status": None, "path": None}
        
        def on_finished(status, path):
            result_container["status"] = status
            result_container["path"] = path
            
        def on_log(msg):
            log_callback(msg)
            
        def on_progress(p, etr):
            progress(p / 100.0, desc=f"Processing... ETR: {etr}")
            
        thread.conversion_finished.connect(on_finished)
        thread.log_updated.connect(on_log)
        thread.progress_updated.connect(on_progress)
        
        # Handle chapter detection dialog (auto-accept defaults)
        # The thread waits on self._chapter_options_event
        # We can start a separate thread to set this event if needed, 
        # but since we mocked QDialog, the dialog logic in ConversionThread won't actually show a GUI.
        # However, ConversionThread logic waits for self._chapter_options_event.wait()
        # We need to ensure that event is set.
        
        # Monkey patch the _chapter_options_event.wait to return immediately or handle it
        # Actually, looking at code:
        # if (is_txt_file and total_chapters > 1 ...):
        #    self.chapters_detected.emit(total_chapters)
        #    self._chapter_options_event.wait()
        
        # We can connect to chapters_detected and set the event
        def on_chapters_detected(count):
            print(f"Chapters detected: {count}")
            # Auto-select defaults
            thread.save_chapters_separately = False
            thread.merge_chapters_at_end = True
            thread._chapter_options_event.set()
            
        thread.chapters_detected.connect(on_chapters_detected)

        # Start the thread
        thread.start()
        
        # Wait for thread to finish (since we are in a Gradio function)
        thread.join()
        
        # Return result
        log_str = "\n".join(logs)
        
        if result_container["path"]:
            return log_str, result_container["path"]
        else:
            return log_str, None
            
    except Exception as e:
        import traceback
        return f"Error: {e}\n{traceback.format_exc()}", None

# --- Gradio Interface ---

with gr.Blocks(title="Abogen WebUI") as app:
    gr.Markdown("# Abogen WebUI")
    gr.Markdown("Generate audiobooks from text or files using Kokoro TTS.")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Input Text", lines=10, placeholder="Paste text here...")
            file_input = gr.File(label="Or Upload File (.txt, .epub, .pdf, .srt, .ass, .vtt)")
            
            with gr.Row():
                language = gr.Dropdown(choices=get_language_choices(), value="a - American English", label="Language")
                voice = gr.Dropdown(choices=get_voice_choices(), value="af_heart", label="Voice")
            
            with gr.Row():
                speed = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Speed")
                output_format = gr.Dropdown(choices=SUPPORTED_SOUND_FORMATS, value="mp3", label="Output Format")
            
            with gr.Row():
                subtitle_mode = gr.Dropdown(choices=["Disabled", "Sentence", "Sentence + Highlighting", "Word"], value="Disabled", label="Subtitle Mode")
                subtitle_format = gr.Dropdown(choices=[f[0] for f in SUPPORTED_SUBTITLE_FORMATS], value="srt", label="Subtitle Format")
            
            split_pattern = gr.Textbox(label="Split Pattern (Regex)", value=r"\n+", visible=False) # Advanced
            
            convert_btn = gr.Button("Generate Audiobook", variant="primary")
        
        with gr.Column():
            logs_output = gr.Textbox(label="Logs", lines=15, interactive=False)
            audio_output = gr.File(label="Download Audiobook")
    
    convert_btn.click(
        fn=run_conversion,
        inputs=[
            text_input,
            file_input,
            language,
            voice,
            speed,
            output_format,
            subtitle_mode,
            subtitle_format,
            split_pattern
        ],
        outputs=[logs_output, audio_output]
    )

if __name__ == "__main__":
    print("Starting Abogen WebUI...")
    app.queue().launch(share=True, server_name="0.0.0.0", server_port=7860)
