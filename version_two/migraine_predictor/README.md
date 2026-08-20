# 🧠 Migraine Predictor System

A digital phenotyping system that monitors laptop activity to predict migraines based on behavioral patterns before symptoms begin.

## Overview

This system quietly monitors natural computer interactions to detect early warning signs of migraines. It observes:

- **Keyboard Activity**: Typing speed, pauses, error frequency, key hold duration
- **Mouse Activity**: Movement patterns, clicking, scrolling, idle periods
- **Webcam/Facial**: Blink rate, head tilt, eye strain, face distance from screen
- **System Usage**: Screen time, app switching, CPU/memory usage, late-night usage

## Features

### Phase 1: Prediction
- Real-time migraine risk assessment (0-100%)
- Risk level classification (Low, Moderate, High, Critical)
- Contributing factor identification
- Historical tracking and trends
- Personalized baseline calibration

### Phase 2: Therapy Recommendations
- Migraine type classification:
  - Tension-type headache
  - Migraine with aura
  - Migraine without aura
  - Cluster headache
- Therapy recommendations:
  - Pressure therapy settings
  - Temperature therapy (cooling/warming)
  - Rest and hydration reminders

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam (optional, for facial monitoring)
- pip package manager

### Setup

1. **Clone/Navigate to the project directory:**
```bash
cd migraine_predictor
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the application:**
```bash
python run.py
```

5. **Open your browser:**
Navigate to `http://localhost:5000`

## Usage

### Starting Monitoring

1. Open the dashboard at `http://localhost:5000`
2. Click "Start Monitoring" to begin collecting data
3. The system will run in the background monitoring your activity

### Dashboard Features

- **Main Dashboard**: Real-time risk score and monitor status
- **Detailed Dashboard**: Live charts and detailed metrics
- **History**: View past predictions and trends
- **Settings**: Configure monitors, alerts, and calibration

### Calibration

For best results, calibrate the system when you're feeling well:

1. Go to Settings
2. Click "Start Calibration"
3. Work normally for 5 minutes
4. The system learns your baseline behavior

### Reporting Migraines

When you experience a migraine:

1. Click "Report Migraine" on the dashboard
2. Select the type and intensity
3. Add any notes
4. This helps improve prediction accuracy

## Project Structure

```
migraine_predictor/
├── run.py                 # Main application entry point
├── app.py                 # Flask app configuration
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
├── monitors/             # Activity monitoring modules
│   ├── __init__.py
│   ├── keyboard_monitor.py
│   ├── mouse_monitor.py
│   ├── webcam_monitor.py
│   └── system_monitor.py
│
├── prediction/           # ML prediction module
│   ├── __init__.py
│   └── migraine_model.py
│
├── models/               # Database models
│   ├── __init__.py
│   └── database.py
│
├── templates/            # HTML templates
│   ├── index.html
│   ├── dashboard.html
│   ├── history.html
│   └── settings.html
│
└── static/               # Static assets
    ├── css/
    │   └── style.css
    └── js/
        └── dashboard.js
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Get monitoring status |
| `/api/start` | POST | Start monitoring |
| `/api/stop` | POST | Stop monitoring |
| `/api/features` | GET | Get current feature values |
| `/api/predict` | GET | Get current prediction |
| `/api/history` | GET | Get prediction history |
| `/api/report_migraine` | POST | Report a migraine |
| `/api/therapy_recommendation` | GET | Get therapy recommendations |

## How It Works

### Data Collection
The system collects behavioral signals from four areas:
- Keyboard patterns indicate cognitive slowdown and mental fatigue
- Mouse patterns reflect attention and physical fatigue
- Facial cues show eye strain and tension
- System usage reveals stress and workload

### Prediction Model
The system uses a combination of:
- Rule-based fatigue scoring for each monitor
- Weighted combination of fatigue indicators
- Trend analysis over time
- Baseline deviation detection

### Risk Levels
- **Low (0-30%)**: Normal activity, no concerns
- **Moderate (30-50%)**: Some fatigue indicators present
- **High (50-70%)**: Multiple warning signs detected
- **Critical (70-100%)**: Immediate preventive action recommended

## Privacy

- ✅ All data is stored locally on your device
- ✅ No data is sent to external servers
- ✅ Webcam images are processed but never saved
- ✅ Keyboard content (what you type) is NOT recorded
- ✅ Only patterns and statistics are tracked

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | auto-generated | Flask secret key |
| `DATABASE_URL` | sqlite:///migraine_data.db | Database connection |

### Monitor Settings

Configure in Settings page or `config.py`:
- Enable/disable individual monitors
- Alert thresholds
- Notification preferences
- Calibration data

## Troubleshooting

### Webcam not working
- Ensure camera permissions are granted
- The system will use simulation mode if camera is unavailable

### High CPU usage
- Reduce webcam frame rate in settings
- Disable webcam monitoring if not needed

### Keyboard/Mouse not tracking
- On macOS: Grant accessibility permissions
- On Linux: May need to run with elevated permissions

## Future Enhancements

- [ ] Machine learning model training with user data
- [ ] Mobile app companion
- [ ] Smart watch integration
- [ ] Weather and environmental factor correlation
- [ ] Medication tracking
- [ ] Export reports for healthcare providers

## License

This project is for educational and research purposes.

## Acknowledgments

Based on research showing that migraines are preceded by subtle behavioral changes hours before symptoms begin, including reduced concentration, increased fatigue, and eye strain.
