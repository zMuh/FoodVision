.PHONY: train tune eval predict help

# defaults (override on make command line)
DATA ?= ./raw_data/food-101/food101_yolo_small
MODEL ?= yolo11n-cls.pt
EPOCHS ?= 10
BATCH ?= 64
DEVICE ?= mps
ITERATIONS ?= 300
OPTIMIZER ?= SGD
WEIGHTS ?= ./models/best.pt
SOURCE ?= ./raw_data/food-101/food101_yolo/val/pad_thai/3828756.jpg
IMGSZ ?= 224
PY ?= python3

train:
	$(PY) model/model.py train --data $(DATA) --model $(MODEL) --epochs $(EPOCHS) --batch $(BATCH) --device $(DEVICE)

tune:
	$(PY) model/model.py tune --data $(DATA) --model $(MODEL) --epochs $(EPOCHS) --iterations $(ITERATIONS) --optimizer $(OPTIMIZER)

eval:
	$(PY) model/model.py eval --weights $(WEIGHTS) --data $(DATA)

predict:
	$(PY) model/model.py predict --weights '$(WEIGHTS)' --source '$(SOURCE)' --imgsz $(IMGSZ)

help:
	@echo "Available targets: train tune eval predict"
	@echo "Examples:"
	@echo "  make predict"
	@echo "  make predict WEIGHTS=./models/my.pt SOURCE=./raw_data/.../image.jpg IMGSZ=224"
# usage Example : make predict WEIGHTS=/Users/muhannad/code/zMuh/FoodVision/models/baselinemodel.pt \
             SOURCE=/Users/muhannad/code/zMuh/FoodVision/raw_data/food-101/food101_yolo/val/pad_thai/3828756.jpg \
             IMGSZ=224
