# ONNX-Runtime examples

## conda setup
```bash
conda env create --file environment-gpu.yml
conda activate onnxruntime-gpu

# run the examples
./simple_onnxruntime_inference.py
./resnet50_modelzoo_onnxruntime_inference.py

conda deactivate
conda env remove -n onnxruntime-gpu
```

## pip Setup
Set python to python3 as default
```bash
sudo ln -sfn /usr/bin/python3 /usr/bin/python
```

Create & activate virtual environment
```bash
pip3 install --user virtualenv
virtualenv venv
source venv/bin/activate
```

Install requirements
```bash
apt install python3-pip
pip install -r requirements.txt
```

