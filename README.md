# Voice Transcription Tool

A simple push-to-talk voice transcription tool that uses Whisper AI to transcribe your speech and automatically paste it wherever your cursor is.

## Features

- Push-to-talk recording (hold Ctrl+Shift to record, release to transcribe)
- Local processing with Whisper Base model (~140MB)
- Automatic clipboard paste to cursor location
- Fast transcription with CPU optimization
- Privacy-focused (everything runs locally)

## Requirements

- macOS
- Python 3.8 or higher
- Microphone access

## Quick Install

1. **Clone or download this repository to `~/Documents/voice`**

2. **Navigate to the directory:**
   ```bash
   cd ~/Documents/voice
   ```

3. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   ```

4. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

5. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   The Whisper base model (~140MB) will be automatically downloaded on first run.

6. **Add the alias to your `.zshrc` (optional but recommended):**

   Add this line to your `~/.zshrc`:
   ```bash
   alias voice='cd ~/Documents/voice && source venv/bin/activate && python voice_transcribe.py'
   ```

   Then reload your shell:
   ```bash
   source ~/.zshrc
   ```

## Usage

**With alias:**
```bash
voice
```

**Without alias:**
```bash
cd ~/Documents/voice
source venv/bin/activate
python voice_transcribe.py
```

### How to Use

1. **Grant permissions when prompted:**
   - Microphone access (System Settings > Privacy & Security > Microphone)
   - Accessibility permissions for auto-paste (System Settings > Privacy & Security > Accessibility)

2. **Recording:**
   - Press and HOLD `Ctrl+Shift` to start recording
   - Speak clearly into your microphone
   - RELEASE either key to stop recording and transcribe
   - The transcribed text will be copied to clipboard and auto-pasted at your cursor

3. **Exit:**
   - Press `ESC` to quit

## Customization

Edit the settings in `voice_transcribe.py` (line 208):

```python
MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large
```

### Model Size Tradeoffs
- **tiny** (~75MB): Fastest, good for clear speech
- **base** (~140MB): Better accuracy, still fast (current default)
- **small** (~460MB): Best balance of speed and accuracy

## Troubleshooting

### "No audio recorded" or silence issues
- Check microphone is working in System Settings
- Speak closer to the microphone
- Increase your microphone volume

### Permission errors
- Go to System Settings > Privacy & Security > Microphone
- Enable access for Terminal (or your terminal app)
- May need to enable Input Monitoring for keyboard access

### Script won't start
- Make sure virtual environment is activated: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

### Paste doesn't work
- Make sure the cursor is in a text field
- Check that clipboard permissions are granted
- Try clicking in the target field before transcribing

## Technical Details

- **Audio**: 16kHz mono, captured via sounddevice
- **Model**: Whisper Tiny with INT8 quantization for CPU
- **Transcription**: Uses VAD (Voice Activity Detection) to filter silence
- **Clipboard**: Uses pyperclip + keyboard automation for pasting

## Privacy

All processing happens locally on your machine. No audio or text is sent to external servers.
