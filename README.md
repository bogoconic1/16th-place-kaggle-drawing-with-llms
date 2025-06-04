# 16th place solution for "Drawing with LLMs"

Team:
- https://www.kaggle.com/yeoyunsianggeremie
- https://www.kaggle.com/xyzdivergence
- https://www.kaggle.com/evgeniimaslov2

Editorial: https://www.kaggle.com/competitions/drawing-with-llms/discussion/581094

In addition, we also deployed our pipeline onto [Modal](https://modal.com/), utilizing FastAPI and Gradio.
The link to access the app is [here](https://geremieyeo--text-to-svg-generator-run-dev.modal.run//)

# Notes
The original OpenAI CLIP repository does not support loading safetensors or bin files so I had to use [my forked repo](https://github.com/bogoconic1/CLIP) to add that functionality. 
