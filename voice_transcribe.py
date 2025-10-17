#!/usr/bin/env python3
"""
Voice transcription tool - Press and hold a key to record, release to transcribe.
The transcribed text is copied to clipboard and automatically pasted.
"""

import sys
import threading
import time
import sounddevice as sd
import numpy as np
import pyperclip
from pynput import keyboard
from faster_whisper import WhisperModel


class VoiceTranscriber:
    def __init__(self, model_size="tiny"):
        """
        Initialize the voice transcriber.

        Args:
            model_size: Whisper model size (tiny, base, small, etc.)
        """
        self.model_size = model_size
        self.is_recording = False
        self.audio_data = []
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.model = None
        self.current_keys = set()
        self.stream = None
        self.stream_error = False

        print(f"Loading Whisper {model_size} model...")
        print("This may take a moment on first run (downloading model)...")

        # Load model with CPU-optimized settings
        # Use int8 for faster inference on CPU
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print(f"Model loaded successfully!")

    def audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio stream - captures audio while recording."""
        if status:
            # Check for critical audio errors (device disconnected after sleep)
            if 'Error' in str(status):
                self.stream_error = True
            print(f"Audio status: {status}", file=sys.stderr)
        if self.is_recording:
            self.audio_data.append(indata.copy())

    def restart_audio_stream(self):
        """Restart the audio stream (useful after laptop wakes from sleep)."""
        try:
            if self.stream:
                print("🔄 Restarting audio stream...")
                self.stream.stop()
                self.stream.close()

            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_callback,
                dtype=np.float32
            )
            self.stream.start()
            self.stream_error = False
            print("✅ Audio stream restarted successfully")
        except Exception as e:
            print(f"❌ Failed to restart audio stream: {e}")
            self.stream_error = True

    def start_recording(self):
        """Start recording audio."""
        if not self.is_recording:
            # Check if stream needs to be restarted
            if self.stream_error or not self.stream or not self.stream.active:
                self.restart_audio_stream()
                if self.stream_error:
                    print("❌ Cannot record - audio stream not available")
                    return

            self.is_recording = True
            self.audio_data = []
            print("\n🎤 Recording... (release key to transcribe)")

    def stop_recording(self):
        """Stop recording and transcribe."""
        if self.is_recording:
            self.is_recording = False
            print("⏹️  Recording stopped. Transcribing...")

            # Process audio in a separate thread to avoid blocking
            threading.Thread(target=self.transcribe_and_paste, daemon=True).start()

    def transcribe_and_paste(self):
        """Transcribe the recorded audio and paste it."""
        try:
            if not self.audio_data:
                print("❌ No audio recorded")
                return

            # Concatenate all audio chunks
            audio = np.concatenate(self.audio_data, axis=0)

            # Convert to mono if stereo
            if audio.shape[1] > 1:
                audio = audio.mean(axis=1)
            else:
                audio = audio.flatten()

            # Normalize audio
            audio = audio.astype(np.float32)

            # Transcribe
            segments, info = self.model.transcribe(
                audio,
                beam_size=1,  # Faster inference with smaller beam
                language="en",  # Set to None for auto-detection
                vad_filter=True,  # Filter out silence
            )

            # Collect all transcribed text
            transcribed_text = " ".join([segment.text for segment in segments]).strip()

            if transcribed_text:
                print(f"✅ Transcribed: {transcribed_text}")

                # Copy to clipboard
                pyperclip.copy(transcribed_text)
                print("📋 Copied to clipboard")

                # Small delay to ensure clipboard is ready
                time.sleep(0.2)

                # Simulate Cmd+V to paste
                try:
                    controller = keyboard.Controller()
                    # Add small delay before paste
                    time.sleep(0.1)
                    # Press cmd
                    controller.press(keyboard.Key.cmd)
                    time.sleep(0.05)
                    # Press v
                    controller.press('v')
                    time.sleep(0.05)
                    # Release v
                    controller.release('v')
                    time.sleep(0.05)
                    # Release cmd
                    controller.release(keyboard.Key.cmd)
                    print("✨ Pasted to cursor location")
                except Exception as e:
                    print(f"⚠️  Could not auto-paste: {e}")
                    print("💡 Text is in clipboard - paste manually with Cmd+V")
            else:
                print("❌ No speech detected")

        except Exception as e:
            print(f"❌ Error during transcription: {e}")

    def on_press(self, key):
        """Handle key press events."""
        try:
            # Add key to current keys
            self.current_keys.add(key)

            # Check for Ctrl+Shift+Esc to exit
            if (keyboard.Key.ctrl_l in self.current_keys or keyboard.Key.ctrl in self.current_keys) and \
               (keyboard.Key.shift_l in self.current_keys or keyboard.Key.shift in self.current_keys) and \
               (key == keyboard.Key.esc):
                print("\n👋 Exiting...")
                return False

            # Check for Ctrl+Shift combination (both pressed together) to record
            if (keyboard.Key.ctrl_l in self.current_keys or keyboard.Key.ctrl in self.current_keys) and \
               (keyboard.Key.shift_l in self.current_keys or keyboard.Key.shift in self.current_keys):
                if not self.is_recording and key != keyboard.Key.esc:
                    self.start_recording()
        except AttributeError:
            pass

    def on_release(self, key):
        """Handle key release events."""
        try:
            # Stop recording when either Ctrl or Shift is released
            if self.is_recording:
                if key in [keyboard.Key.ctrl_l, keyboard.Key.ctrl, keyboard.Key.shift_l, keyboard.Key.shift]:
                    self.stop_recording()

            # Remove key from current keys
            if key in self.current_keys:
                self.current_keys.remove(key)
        except AttributeError:
            pass

    def run(self):
        """Main loop - listen for hotkey and manage recording."""
        print("\n" + "="*60)
        print("🎙️  Voice Transcription Tool")
        print("="*60)
        print("Hotkey: Ctrl+Shift")
        print("Press and HOLD Ctrl+Shift to record")
        print("RELEASE either key to stop recording and transcribe")
        print("Press Ctrl+Shift+Esc to quit")
        print("="*60 + "\n")

        # Open audio stream
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_callback,
                dtype=np.float32
            )

            self.stream.start()
            print("✅ Ready! Waiting for hotkey...")

            # Set up keyboard listener
            with keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release
            ) as listener:
                listener.join()

            self.stream.stop()
            self.stream.close()
            print("\n\n👋 Shutting down...")

        except KeyboardInterrupt:
            print("\n\n👋 Shutting down...")
            if self.stream:
                self.stream.stop()
                self.stream.close()
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure you've granted microphone permissions")
            print("2. Check that no other app is using the microphone")
            print("3. Try running with 'sudo' if you get permission errors")
            if self.stream:
                self.stream.stop()
                self.stream.close()


def main():
    """Entry point for the application."""
    # You can customize this setting
    MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large

    try:
        transcriber = VoiceTranscriber(model_size=MODEL_SIZE)
        transcriber.run()
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
