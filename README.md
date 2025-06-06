## 16th place solution for "Drawing with LLMs"

Team:
- https://www.kaggle.com/yeoyunsianggeremie
- https://www.kaggle.com/xyzdivergence
- https://www.kaggle.com/evgeniimaslov2

Editorial: https://www.kaggle.com/competitions/drawing-with-llms/discussion/581094

In addition, we also deployed our pipeline onto [Modal](https://modal.com/), utilizing FastAPI and Gradio.

Link to access the app: [https://geremieyeo--text-to-svg-generator-run-dev.modal.run//](https://geremieyeo--text-to-svg-generator-run.modal.run//)

## Directory Structure

```
├── backend.py           # Modal deployment configuration with Gradio
├── code/
│   ├── svg_generator.py # Core SVG generation logic
│   └── metric.py        # Scoring and evaluation metrics
├── conf/
│   └── config.yaml      # Model configuration
├── requirements.txt     # Python dependencies
├── setup.sh            # Setup script
├── LICENSE             # License file
└── README.md           # This file
```

## Notes
The original OpenAI CLIP repository does not support loading safetensors or bin files so I had to use [my forked repo](https://github.com/bogoconic1/CLIP) to add that functionality. 
